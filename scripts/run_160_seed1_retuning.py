#!/usr/bin/env python3
"""Fresh225-grid seed1 retuning for160/23, then a matched-partition160/22 check."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE), str(CODE / 'scripts/molecular_figures/analysis')]
import cca401_pipeline as pipeline
from cca401 import nested as run, seed_screen as shared
from cca401_union_grid import GRID, PAIRS

ROOTS = {
    23: CODE / 'results/cca401_160x23_seed1_retuned_mean_tminus1_20261031',
    22: CODE / 'results/cca401_160x22_seed1_matched_splits_20261031',
}
ASSETS = CODE / 'results/baseline_shared_assets_20261031'
CASE = 23
OUT = ROOTS[CASE]
SCALES = run.SCALES_23


def configure(count):
    global CASE, OUT, SCALES
    CASE, OUT = count, ROOTS[count]
    SCALES = run.SCALES_23 if count == 23 else run.SCALES_22_WITHOUT_HLP
    pipeline.CURRENT, pipeline.SCALES = OUT, SCALES
    pipeline.PAIRS = shared.PAIRS = PAIRS
    shared.VOLUME_POLICY = 'both'
    pipeline.load_data = load_data
    pipeline.select_grid = select_full_grid


def load_data():
    meta = json.loads((OUT / 'inputs/features.json').read_text())
    assert meta['status'] == 'complete' and meta['n_patients'] == 160
    assert meta['n_scales'] == CASE and tuple(meta['scales']) == tuple(SCALES)
    assert meta['covariance_denominator'] == 'T-1'
    assert meta['aggregation'] == 'arithmetic_mean_fisher_z'
    with np.load(OUT / 'inputs/features.npz') as z:
        ids, y, volume = z['ids'], z['y'], z['volume']
        pack = {name: z[name].astype(np.float32) for name in ('sc_nc', 'fc')}
    assert len(ids) == 160 and y.shape == (160, CASE)
    assert np.isfinite(y).all() and not (y == 999).any()
    return ids, pack, y, volume


def select_full_grid(x, y, volume, model, policy, seed, k):
    assert policy == 'both' and seed == 1 and k == 5
    with np.load(OUT / 'inputs/split_plan.npz') as z:
        splits = [(z[f'full_train_{j}'], z[f'full_test_{j}']) for j in range(1, 6)]
    return pipeline.select_grid_from_splits(x, y, volume, model, policy, seed, k, splits)




def fit_model(model):
    ids, pack, y, volume = load_data()
    x = run.select_features(pack, model)
    cfg = shared.config_for(model, 1, 10, 5, SCALES)
    folder = OUT / f'oof/both/{model.replace("+", "_")}/10x5/seed_1'
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(OUT / 'inputs/split_plan.npz') as z:
        outer = [(z[f'outer_train_{k}'], z[f'outer_test_{k}']) for k in range(1, 11)]
        for k in range(1, 11):
            path = folder / f'fold_{k:02d}.npz'
            if path.exists():
                with np.load(path) as saved:
                    np.testing.assert_array_equal(saved['train_index'], outer[k-1][0])
                    np.testing.assert_array_equal(saved['test_index'], outer[k-1][1])
                    np.testing.assert_array_equal(saved['pairs'], PAIRS)
                continue
            inner = [(z[f'inner_train_{k}_{j}'], z[f'inner_test_{k}_{j}']) for j in range(1, 6)]
            result = shared.fold_from_splits(x, y, volume, outer, k, cfg, inner)
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **result)
            temporary.replace(path)
            print(f'{CASE} scales {model} outer fold{k}/10 complete', flush=True)
    shared.DATA = ids, pack, y, volume
    rows = shared.aggregate(folder, model, cfg, SCALES)
    pipeline.write_json(folder / 'manifest.json', dict(
        status='complete', source_kind='fresh_nested_oof', n=160, scales=list(SCALES),
        model=model, seed=1, design='10x5', volume_policy='both', grid_pairs=225,
        reused_grid_pairs=0, rcca_permutations=0,
        partition_source='saved original23-scale seed1 outer and inner partitions'))
    selection = pipeline.full_task((model, 'both', 1, 5))
    selection.update(n_scales=CASE, partition_source='saved original23-scale seed1 full-data5fold',
                     new_parameter_selection=True)
    selection_path = OUT / f'selection/both/{model.replace("+", "_")}/seed_1_5fold/selection.json'
    pipeline.write_json(selection_path, selection)
    choices = {choice['branch']: choice for choice in selection['choices']}
    for row in rows:
        choice = choices[row['branch']]
        row.update(fit_directory=choice['fit_directory'], final_tau_x=choice['tau'][0],
                   final_tau_y=choice['tau'][1], n_scales=CASE,
                   partition_policy='native23' if CASE == 23 else 'matched23')
    print(f'{CASE} scales {model}: nested and full-data selection complete', flush=True)
    return rows


def run_case(count):
    configure(count)
    started = time.monotonic()
    pipeline.write_json(OUT / 'status.json', dict(status='running', phase='retuning', n_scales=CASE))
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn'),
                             initializer=configure, initargs=(CASE,)) as pool:
        jobs = {pool.submit(fit_model, model): model for model in ('nc', 'fc', 'nc+fc')}
        for future in as_completed(jobs):
            model = jobs[future]
            try:
                rows.extend(future.result())
            except Exception:
                failures.append(dict(model=model, traceback=traceback.format_exc()))
            pipeline.write_json(OUT / 'progress.json', dict(
                completed_models=len(rows)//4, failures=failures, elapsed_seconds=time.monotonic()-started))
    if failures:
        raise RuntimeError(f'{len(failures)} model jobs failed; see progress.json')
    clinical = pd.DataFrame(rows).sort_values(['model', 'branch'])
    assert len(clinical) == 12
    clinical.to_csv(OUT / 'clinical_results.csv', index=False)
    primary = clinical.loc[clinical.branch.eq('B')]
    primary.to_csv(OUT / 'primary_clinical.csv', index=False)
    fit = primary.loc[primary.model.eq('nc+fc'), 'fit_directory'].item()
    pipeline.write_json(OUT / 'status.json', dict(status='running', phase='molecular', fit=fit))
    assets = OUT / 'assets'
    if not assets.exists():
        assets.symlink_to(ASSETS, target_is_directory=True)
    import current_molecular as molecular
    molecular.CURRENT, molecular.N_SPIN, molecular.N_BOOT, molecular.SEED = OUT, 100000, 1000, 42
    result = molecular.molecular_task(fit)
    pipeline.write_json(OUT / 'status.json', dict(
        status='complete', n=160, n_scales=CASE, seed=1, clinical_rows=12,
        primary_rows=3, grid_pairs=225, reused_grid_pairs=0, rcca_permutations=0,
        molecular=result, elapsed_seconds=time.monotonic()-started,
        python=platform.python_version(),
        packages={name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'cca-zoo', 'pandas')}))
    print(f'COMPLETE {CASE} scales: {time.monotonic()-started:.1f}s', flush=True)


def export_small(count):
    """Stage final fits and review tables, omitting grid arrays, images and nulls."""
    configure(count)
    assert json.loads((OUT / 'status.json').read_text())['status'] == 'complete'
    export = OUT / 'review_export'
    export.mkdir()
    for name in ('status.json', 'protocol.json', 'progress.json', 'clinical_results.csv',
                 'primary_clinical.csv', 'run.log'):
        shutil.copy2(OUT / name, export / name)
    primary = pd.read_csv(OUT / 'primary_clinical.csv')
    for row in primary.to_dict('records'):
        key = row['model'].replace('+', '_')
        paths = {
            f'oof/both/{key}/10x5/seed_1':
                ('summary.csv', 'selected_parameters.csv', 'outer_scores.csv', 'B_aligned.npz', 'manifest.json'),
            f'selection/both/{key}/seed_1_5fold': ('selection.json', 'grid.csv'),
            row['fit_directory']:
                ('model.npz', 'manifest.json', 'symptom_loadings.csv', 'maps116.csv', 'top20_regions.csv'),
        }
        if row['model'] == 'nc+fc':
            paths[str(Path('molecular') / Path(row['fit_directory']).relative_to('fits'))] = (
                'completion.json', 'pls.csv', 'pet.csv', 'genes.csv', 'enrichment_summary.csv', 'fdc_overlap.json')
        for relative, names in paths.items():
            target = export / relative
            target.mkdir(parents=True)
            for name in names:
                shutil.copy2(OUT / relative / name, target / name)
    print(f'Small review export: {export}', flush=True)








def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'export'))
    parser.add_argument('--scales', type=int, choices=(23, 22), default=23)
    args = parser.parse_args()
    if args.action == 'export':
        export_small(args.scales)
    else:
        try:
            run_case(args.scales)
        except Exception:
            pipeline.write_json(ROOTS[args.scales] / 'status.json', dict(
                status='failed', traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
