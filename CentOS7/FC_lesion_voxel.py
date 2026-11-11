#!/usr/bin/env python

import os, glob, time, socket
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from datetime import timedelta
import multiprocessing as mp
from tqdm import tqdm

# ======================================================
# ENVIRONMENT SETUP
# ======================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


# ======================================================
# PATHS
# ======================================================
atlas = '/Users/yojiao/SFC_project/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_MNI152NLin6Asym_2mm.nii.gz'
brain_mask_path = '/Users/yojiao/SFC_project/MNI152_T1_2mm_brain_mask.nii'
LESION_DIR = '/Shared/boeslab/Data/Lesion/Iowa/Registry/lesionMasks/Chronic/MNI152/2mm'
OUT_ROOT = '/Users/yojiao/SFC_project/LNM_voxel/Iowa_fLNM'
os.makedirs(OUT_ROOT, exist_ok=True)

# GSP dataset
gsp_1000_dataset = '/Shared/jianglab/3_Data_Working/GSP1000_v2_dataset'
gsp_func_paths = sorted(glob.glob(f'{gsp_1000_dataset}/*/func/sub-*_bld_skip4*.nii.gz'))
TOTAL_SUBJ = len(gsp_func_paths)


# ======================================================
# LOAD ATLAS & BRAIN MASK
# ======================================================
atlas_img = nib.load(atlas)
atlas_data = atlas_img.get_fdata().astype(int)

brain_mask_img = nib.load(brain_mask_path)
brain_mask = brain_mask_img.get_fdata() > 0 

# atlas ROI list
roi_ids = np.unique(atlas_data)
roi_ids = roi_ids[roi_ids > 0]
ROI_COUNT = len(roi_ids)


# ======================================================
# IOWA MMPI FILTER
# ======================================================
IOWA_MMPI_LIST = '/Users/yojiao/SFC_project/LNM_voxel/Iowa_Lesion_MMPI.txt'
if not os.path.isfile(IOWA_MMPI_LIST):
    print(f"[MMPIFilter] Missing {IOWA_MMPI_LIST}, exit.")
    raise SystemExit(0)

with open(IOWA_MMPI_LIST, 'r') as f:
    target_ids = {line.strip() for line in f if line.strip()}

def _stem(p):
    b = os.path.basename(p)
    if b.endswith('.nii.gz'): return b[:-7]
    if b.endswith('.nii'): return b[:-4]
    return os.path.splitext(b)[0]

lesion_paths = [p for p in sorted(glob.glob(f'{LESION_DIR}/*.nii*'))
                if _stem(p) in target_ids]

if len(lesion_paths) == 0:
    print("[MMPIFilter] No matched lesions. Exit.")
    raise SystemExit(0)

# ======================================================
# HPC ARRAY INDEX SUPPORT
# ======================================================
arr_sge = os.environ.get('SGE_TASK_ID')
arr_pbs = os.environ.get('PBS_ARRAYID')
arr_man = os.environ.get('LESION_INDEX')
arr_id = arr_sge or arr_pbs or arr_man

if arr_id is not None:
    idx = int(arr_id) - (1 if arr_sge else 0)
    if 0 <= idx < len(lesion_paths):
        lesion_paths = [lesion_paths[idx]]
    else:
        print("[Array] Index out of range. Exit.")
        raise SystemExit(0)

print(f"[Init] host={socket.gethostname()} pid={os.getpid()} | subjects={TOTAL_SUBJ} | ROI={ROI_COUNT}")


# ======================================================
# PARALLEL SETUP
# ======================================================
def _auto_procs(default=4):
    for k in ("N_SUBJ_PROCS", "NSLOTS", "SLURM_CPUS_PER_TASK"):
        v = os.environ.get(k)
        if v and v.isdigit():
            return max(1, int(v))
    return default

N_PROCS = _auto_procs(4)


# ======================================================
# WORKER INITIALIZER (builds maskers once per process)
# ======================================================
_MASKER_BRAIN = None
_MASKER_SEED = None

def _proc_init(lesion_path):
    global _MASKER_BRAIN, _MASKER_SEED

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # voxel-level masker (whole brain)
    _MASKER_BRAIN = NiftiMasker(mask_img=brain_mask_img, standardize="zscore_sample")
    _MASKER_BRAIN.fit()

    # seed masker (lesion mask)
    lesion_img = nib.load(lesion_path)
    _MASKER_SEED = NiftiLabelsMasker(labels_img=lesion_img, standardize="zscore_sample")
    _MASKER_SEED.fit()


# ======================================================
# WORKER: compute Method-B FC vector
# ======================================================
def _worker(args):
    i, func_path = args
    img = nib.load(func_path)

    seed_ts = _MASKER_SEED.transform(img)           # (T,1)
    brain_ts = _MASKER_BRAIN.transform(img)         # (T,V)

    T = seed_ts.shape[0]
    # Both maskers use sample SD (ddof=1), so Pearson covariance divides by T-1.
    fc_vox = (brain_ts.T @ seed_ts) / (T - 1)       # (V,1)
    return i, fc_vox.ravel()


# ======================================================
# MAIN LOOP: each lesion
# ======================================================
t_all = time.time()

for lesion_path in tqdm(lesion_paths, desc="lesions"):
    lesion_id = _stem(lesion_path)

    print(f"\n[Lesion] {lesion_id} | N={TOTAL_SUBJ} | jobs={N_PROCS}", flush=True)

    # Prepare voxel index mapping once (same for all subjects)
    atlas_flat = atlas_data[brain_mask]    # (V_mask,)

    # Output array
    fc_all = [None] * TOTAL_SUBJ

    # Parallel pool
    with mp.get_context("fork").Pool(
        processes=N_PROCS,
        initializer=_proc_init,
        initargs=(lesion_path,)
    ) as pool:

        for i, fc_vec in pool.imap_unordered(_worker, list(enumerate(gsp_func_paths))):
            fc_all[i] = fc_vec

    # Stack into (N × V_mask)
    fc_all = np.vstack(fc_all)

    # ROI averaging
    feats = np.zeros((TOTAL_SUBJ, ROI_COUNT), dtype=np.float32)

    for j, rid in enumerate(roi_ids):
        idx = np.where(atlas_flat == rid)[0]
        if idx.size == 0:
            feats[:, j] = np.nan
        else:
            feats[:, j] = fc_all[:, idx].mean(axis=1)

    # Fisher Z
    feats_z = np.arctanh(feats)

    # ======================================================
    # SHAPE VALIDATION 
    # ======================================================
    if feats_z.shape[0] != TOTAL_SUBJ:
        raise RuntimeError(f"[ERROR] subject count mismatch: {feats_z.shape[0]} vs {TOTAL_SUBJ}")

    if feats_z.shape[1] != ROI_COUNT:
        raise RuntimeError(f"[ERROR] ROI count mismatch: {feats_z.shape[1]} vs {ROI_COUNT}")

    # SAVE
    out_path = os.path.join(OUT_ROOT, f'{lesion_id}.npy')
    np.save(out_path, feats_z)

    print(f"[Save] {lesion_id} -> {feats_z.shape} saved to {out_path}", flush=True)


elapsed = timedelta(seconds=int(time.time() - t_all))
print(f"\n[Summary] Lesions={len(lesion_paths)} | Subjects={TOTAL_SUBJ} | ROI={ROI_COUNT}")
print(f"Total time = {elapsed}")
