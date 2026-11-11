#!/usr/bin/env python3
"""Recompute the same FDC definition from known-T GSP time series, all238 masks.

Factor only the linear regional averaging. Seeds remain the mean raw lesion
signal, standardized with sample SD. Regional brain signals are means of
voxelwise sample-standardized signals, not standardized regional averages.
Native voxelwise calculations validate this factorization before batch use.
All images and per-reference arrays stay on Argon.
"""
from pathlib import Path
import argparse
import csv
import gc
import json
import os
import time

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from nilearn.image import concat_imgs
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.signal import clean

PROJECT = Path('/Users/yojiao/SFC_project')
GSP = Path('/Shared/boeslab/Data/fcMRI/GSP1000')
LESIONS = Path('/Shared/boeslab/Data/Lesion/Iowa/Registry/lesionMasks/Chronic/MNI152/2mm')
ATLAS_NAME = 'Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_MNI152NLin6Asym_2mm.nii.gz'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare(cohort):
    table = pd.read_csv(cohort)
    ids = np.sort(table.loc[table.ID != 2549, 'ID'].to_numpy(int))
    assert len(ids) == len(np.unique(ids)) == 238
    atlas_img = nib.load(PROJECT / ATLAS_NAME)
    atlas = atlas_img.get_fdata().astype(int)
    brain_mask_img = nib.load(PROJECT / 'MNI152_T1_2mm_brain_mask.nii')
    brain_mask = brain_mask_img.get_fdata() > 0
    assert atlas.shape == brain_mask.shape == (91, 109, 91)
    positions, row_indices, counts = [], [], []
    for i, sid in enumerate(ids):
        img = nib.load(LESIONS / f'{sid:04d}.nii.gz')
        assert img.shape == atlas.shape
        np.testing.assert_allclose(img.affine, atlas_img.affine, rtol=0, atol=1e-5)
        mask = img.get_fdata()
        assert np.isin(mask, (0, 1)).all()
        index = np.flatnonzero(mask.reshape(-1))
        assert len(index) > 0
        positions.append(index)
        row_indices.append(np.full(len(index), i, dtype=np.int32))
        counts.append(len(index))
    positions = np.concatenate(positions)
    seeds = csr_matrix((np.ones(len(positions)), (np.concatenate(row_indices), positions)),
                       shape=(238, atlas.size))
    brain_labels = atlas[brain_mask]
    roi_indices = [np.flatnonzero(brain_labels == i) for i in range(1, 117)]
    assert all(len(index) > 0 for index in roi_indices)
    rpos = np.concatenate(roi_indices)
    roi_rows = np.concatenate([np.full(len(index), i) for i, index in enumerate(roi_indices)])
    roi = csr_matrix((np.ones(len(rpos)), (roi_rows, rpos)), shape=(116, brain_mask.sum()))
    with (GSP / 'participants.tsv').open() as stream:
        participants = sorted(csv.DictReader(stream, delimiter='\t'), key=lambda r: r['participant_id'])
    assert len(participants) == 1000
    masker = NiftiMasker(mask_img=brain_mask_img, standardize='zscore_sample').fit()
    return dict(ids=ids, seeds=seeds, seed_counts=np.array(counts), roi=roi,
                roi_counts=np.array([len(x) for x in roi_indices]), roi_indices=roi_indices,
                masker=masker, atlas_img=atlas_img, participants=participants)


def calculate(context, row, validate=False):
    person = context['participants'][row]
    paths = sorted((GSP / person['participant_id'] / 'func').glob('*skip4*.nii.gz'))
    assert len(paths) == int(person['numruns']) and len(paths) in (1, 2)
    images = [nib.load(path) for path in paths]
    for img in images:
        assert img.shape[:3] == context['atlas_img'].shape and img.shape[3] == 120
        np.testing.assert_allclose(img.affine, context['atlas_img'].affine, rtol=0, atol=1e-5)
    image = concat_imgs(images) if len(images) == 2 else images[0]
    T = image.shape[3]
    raw = np.asarray(image.dataobj, dtype=np.float32)
    assert np.isfinite(raw).all()
    # Avoid simultaneously copying the full4D image into C order and float64.
    # Each time point has exactly the same voxelwise sparse sum as before.
    seed_means = np.empty((238, T), dtype=np.float32)
    for start in range(0, T, 16):
        stop = min(start + 16, T)
        block = raw[..., start:stop].reshape(-1, stop - start)
        seed_means[:, start:stop] = (context['seeds'] @ block) / context['seed_counts'][:, None]
        del block
    seed_z = clean(seed_means.T.astype(np.float32), detrend=False,
                   standardize='zscore_sample', ensure_finite=True)
    del raw, seed_means
    brain_z = context['masker'].transform(image)
    region_means = (context['roi'] @ brain_z.T) / context['roi_counts'][:, None]
    r = seed_z.astype(float).T @ region_means.T / (T - 1)
    assert r.shape == (238, 116) and np.isfinite(r).all() and (abs(r) < 1).all()
    z = np.arctanh(r.astype(np.float32))
    checks = []
    if validate:
        # Anchor, one new patient, and the smallest/largest actual lesion masks.
        selected = sorted(set([int(np.flatnonzero(context['ids'] == 46)[0]),
                               int(np.flatnonzero(context['ids'] == 414)[0]),
                               int(np.argmin(context['seed_counts'])),
                               int(np.argmax(context['seed_counts']))]))
        for i in selected:
            sid = int(context['ids'][i])
            native_seed = NiftiLabelsMasker(labels_img=str(LESIONS / f'{sid:04d}.nii.gz'),
                                            standardize='zscore_sample').fit_transform(image)
            seed_error = float(abs(native_seed[:, 0] - seed_z[:, i]).max())
            voxel_r = (brain_z.T @ native_seed).ravel() / (T - 1)
            native_r = np.array([voxel_r[index].mean() for index in context['roi_indices']], dtype=np.float32)
            native_z = np.arctanh(native_r)
            error = float(abs(z[i] - native_z).max())
            assert seed_error < 2e-6 and error < 2e-6, (row, sid, seed_error, error)
            checks.append(dict(study_id=sid, seed_max_abs_error=seed_error,
                               regional_z_max_abs_error=error))
    record = dict(row=row, subject=person['participant_id'], T=int(T), n_runs=len(paths),
                  known_T_source='actual input image frame count', n_patients=238,
                  covariance_denominator='T-1', source_files=[str(p) for p in paths],
                  finite=True, native_checks=checks)
    del image, images, brain_z, region_means, seed_z
    gc.collect()
    return z, record


def save_row(out, ids, row, z, record):
    folder = out / 'rows'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'row_{row:04d}.npz'
    temp = target.with_suffix('.tmp.npz')
    write_json(folder / f'row_{row:04d}.json', record)
    np.savez_compressed(temp, ids=ids, z=z, row=row, subject=record['subject'], T=record['T'])
    temp.replace(target)


def collect(args):
    acceptance = json.loads((args.out / 'native_validation.json').read_text())
    assert acceptance['status'] == 'passed'
    ids = None
    total = np.zeros((238, 116), dtype=float)
    rows = []
    for row in range(1000):
        record = json.loads((args.out / f'rows/row_{row:04d}.json').read_text())
        assert record['row'] == row and record['finite'] and record['T'] in (120, 240)
        with np.load(args.out / f'rows/row_{row:04d}.npz') as z:
            if row == 0:
                ids = z['ids'].copy()
            np.testing.assert_array_equal(z['ids'], ids)
            assert str(z['subject']) == record['subject'] and int(z['T']) == record['T']
            assert z['z'].shape == total.shape and np.isfinite(z['z']).all()
            total += z['z']
        rows.append(record)
    assert len({r['subject'] for r in rows}) == 1000
    np.savez_compressed(args.out / 'fdc238.npz', ids=ids, fc=total / 1000)
    meta = dict(status='complete', n_patients=238, n_reference_people=1000,
                covariance_denominator='T-1', aggregation='arithmetic_mean_fisher_z',
                healthy_reference='same1000 GSP participants, currently available Boes source runs',
                n_regions=116, historical_v2_T_recovered=False, historical_arrays_reused=False,
                native_validation=acceptance)
    write_json(args.out / 'fdc238.json', meta)
    audit = dict(status='passed', checked_participants=1000,
                 method='direct_recompute_with_known_actual_T', historical_v2_T_recovered=False,
                 historical_replay_difference='separate unresolved historical provenance diagnostic',
                 T_counts={str(T):sum(r['T']==T for r in rows) for T in (120, 240)},
                 aggregation='arithmetic_mean_fisher_z', covariance_denominator='T-1',
                 native_validation=acceptance, image_downloads=0)
    write_json(args.out / 'fresh_source_audit.json', audit)
    pd.DataFrame([{k:r[k] for k in ('row','subject','T','n_runs')} for r in rows]).to_csv(
        args.out / 'reference_timepoints.csv', index=False)
    print(json.dumps(audit, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('validate', 'batch', 'collect'), required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'collect':
        collect(args)
        return
    context = prepare(args.cohort)
    if args.mode == 'validate':
        all_checks = []
        for row in (0, 1, 141):
            started = time.monotonic()
            z, record = calculate(context, row, validate=True)
            existing = args.out / f'rows/row_{row:04d}.npz'
            if existing.exists():
                with np.load(existing) as old:
                    np.testing.assert_array_equal(z, old['z'])
                record['prior_factorization_array_equal'] = True
            save_row(args.out, context['ids'], row, z, record)
            all_checks.append(record)
            print(json.dumps(dict(row=row, seconds=time.monotonic()-started,
                                  native_checks=record['native_checks'])), flush=True)
        write_json(args.out / 'native_validation.json', dict(status='passed',
                   source='native sample-standardized voxelwise source formula',
                   max_allowed_absolute_error=2e-6, rows=all_checks, n_patients=238,
                   covariance_denominator='T-1', historical_array_equivalence_claimed=False))
        return
    assert json.loads((args.out / 'native_validation.json').read_text())['status'] == 'passed'
    task = int(os.environ['SGE_TASK_ID'])
    assert 1 <= task <= 40
    for row in range((task-1)*25, task*25):
        path = args.out / f'rows/row_{row:04d}.npz'
        if path.exists():
            with np.load(path) as z:
                np.testing.assert_array_equal(z['ids'], context['ids'])
                assert str(z['subject']) == context['participants'][row]['participant_id']
            assert json.loads(path.with_suffix('.json').read_text())['row'] == row
            continue
        started = time.monotonic()
        z, record = calculate(context, row)
        save_row(args.out, context['ids'], row, z, record)
        print(json.dumps(dict(row=row, subject=record['subject'], T=record['T'],
                              seconds=time.monotonic()-started)), flush=True)
    write_json(args.out / f'batch_{task:02d}.json', dict(status='complete', task=task))


if __name__ == '__main__':
    main()
