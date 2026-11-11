#!/usr/bin/env python3
# FC_GSP.py
# Build subject-by-feature FC matrix from GSP only (no lesion), with parallel extraction and single-shot correlation.
# =======================================================================
# NOTE: Special handling for subject index 13
# -----------------------------------------------------------------------
# For subject #13 in func_imgs, some timepoints contain NaN. Before
# calling masker.transform, those NaN timepoints are removed. All other
# subjects are unchanged. If no valid timepoints remain after removal,
# a RuntimeError is raised to avoid silent failures.
# =======================================================================

import os, glob, time
import numpy as np
import nibabel as nib
from tqdm import tqdm
from datetime import timedelta
from joblib import Parallel, delayed
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

# ---- BLAS guard ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---- Config ----
N_JOBS = 64
TEST_N_SUBJECTS = None
OUT_DIR = "/home/yojiao/hpchome/SFC_project/GSP_FC"
OUT_FILE = os.path.join(OUT_DIR, "GSP_FC_Schaefer116.npy")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Paths ----
atlas = '/home/yojiao/sfc/atlas/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_MNI152NLin6Asym_2mm.nii.gz'
BRAIN_MASK = "/home/yojiao/LSS/jianglab/3_Data_Working/masks/MNI152_T1_2mm_brain_mask.nii"
GSP_ROOT = "/home/yojiao/LSS/jianglab/3_Data_Working/GSP1000_v2_dataset"

# ---- Subjects ----
func_imgs = sorted(glob.glob(f"{GSP_ROOT}/*/func/sub-*_bld_skip4*.nii.gz"))
if TEST_N_SUBJECTS is not None:
    func_imgs = func_imgs[:TEST_N_SUBJECTS]
if not func_imgs:
    raise RuntimeError("No GSP functional images found.")

# ---- Masker ----
bn_img = nib.load(atlas)
brain_img = nib.load(BRAIN_MASK)
masker = NiftiLabelsMasker(labels_img=bn_img, mask_img=brain_img, standardize="zscore_sample")
masker.fit()

# ---- Parallel extraction (subject #13 NaN removal only) ----
def _transform_one(i, pth):
    import warnings
    warnings.filterwarnings("ignore")
    img = nib.load(pth)
    if i == 13:
        data = img.get_fdata()                       # (X,Y,Z,T)
        bad_tp = np.isnan(data).any(axis=(0, 1, 2))  # (T,)
        good_tp = ~bad_tp
        clean = data[..., good_tp]
        img = nib.Nifti1Image(clean, img.affine, img.header)
    return masker.transform(img)  # (T, R)

t0 = time.time()
sigs = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(_transform_one)(i, pth) for i, pth in tqdm(enumerate(func_imgs), total=len(func_imgs), desc="Extract", ncols=80)
)

# ---- QC (read-only) ----
nonfinite_sigs = sum((~np.isfinite(s)).sum() for s in sigs)
zv_counts = [int((np.nanstd(s, axis=0) == 0).sum()) for s in sigs]
print(f"QC sigs nonfinite: {nonfinite_sigs}")
print(f"QC zero-variance subjects: {sum(v>0 for v in zv_counts)}/{len(sigs)}  total zero-variance cols: {sum(zv_counts)}")

# ---- Correlation (single shot) ----
corr = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
feats = corr.fit_transform(tqdm(sigs, desc="Corr(all)", ncols=80))  # (n_subj, n_feat)

# ---- Fisher z ----
Fz = np.arctanh(feats)
nonfinite_Fz = int((~np.isfinite(Fz)).sum())
print(f"QC Fz nonfinite: {nonfinite_Fz}")

# ---- Save ----
np.save(OUT_FILE, Fz.astype(np.float32))
print(f"Saved: {OUT_FILE}  shape={Fz.shape}")
print(f"Time: {timedelta(seconds=int(time.time() - t0))}")
