# === Fast FC with subject-outer loop (subjects-parallel, minimal changes, lesion-level error handling) ===
import os, glob, time
import numpy as np
import nibabel as nib
from tqdm import tqdm
from nilearn import image
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
from datetime import timedelta
from joblib import Parallel, delayed

# Optional: avoid BLAS oversubscription when using many processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---- Parallel config ----
N_JOBS = 64                 # number of workers for subjects
TEST_N_LESIONS = None        # set small numbers to dry-run; None for all
TEST_N_SUBJECTS = None       # set small numbers to dry-run; None for all

# ---- Paths ----
# atlas = '/home/yojiao/LSS/jianglab/3_Data_Working/masks/BN_Atlas_246_2mm.nii.gz'
atlas = '/home/yojiao/sfc/atlas/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_MNI152NLin6Asym_2mm.nii.gz'
BRAIN_MASK = '/home/yojiao/LSS/jianglab/3_Data_Working/masks/MNI152_T1_2mm_brain_mask.nii'
LESION_DIR = '/home/yojiao/LSS/boeslab/Data/Lesion/Iowa/Registry/lesionMasks/Chronic/MNI152/2mm'
OUT_ROOT = '/home/yojiao/hpchome/SFC_project/GSP_FC_Lesion_Iowa'
os.makedirs(OUT_ROOT, exist_ok=True)
LOG_FILE = os.path.join(OUT_ROOT, "lesions_failed.txt")

# ---- GSP ----
gsp_1000_dataset = '/home/yojiao/LSS/jianglab/3_Data_Working/GSP1000_v2_dataset/'
gsp_1000_func_imgs = sorted(glob.glob(f'{gsp_1000_dataset}/*/func/sub-*_bld_skip4*.nii.gz'))
if TEST_N_SUBJECTS is not None:
    gsp_1000_func_imgs = gsp_1000_func_imgs[:TEST_N_SUBJECTS]

# ---- Load atlas & brain mask ----
bn_img = nib.load(atlas)
bn_data = bn_img.get_fdata()
brain_img = nib.load(BRAIN_MASK)

# ---- Corr measure ----
corr = ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)

# ---- Lesions ----
lesion_paths = sorted(glob.glob(f'{LESION_DIR}/*.nii*'))
if TEST_N_LESIONS is not None:
    lesion_paths = lesion_paths[:TEST_N_LESIONS]

def _transform_one_subject(masker, pth):
    import warnings
    warnings.filterwarnings("ignore")
    img = nib.load(pth)
    return masker.transform(img)

t0 = time.time()
no_overlap = []

for p in tqdm(lesion_paths, desc='lesions'):
    try:
        lesion_sub = os.path.basename(p).replace('.nii.gz','').replace('.nii','')

        lesion_img = nib.load(p)
        lesion_bin = lesion_img.get_fdata() > 0
        if not np.any(lesion_bin & (bn_data > 0)):
            no_overlap.append(lesion_sub)
            continue

        mask_noLesion = image.math_img('img1 - img2', img1=brain_img, img2=lesion_img)

        masker = NiftiLabelsMasker(labels_img=bn_img,
                                   standardize="zscore_sample",
                                   mask_img=mask_noLesion)
        masker.fit()

        # subjects-parallel hotspot
        sigs = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_transform_one_subject)(masker, pth) for pth in gsp_1000_func_imgs
        )

        feats = corr.fit_transform(sigs)
        feats = np.arctanh(feats)
        np.save(os.path.join(OUT_ROOT, f'{lesion_sub}.npy'), feats)

    except Exception as e:
        with open(LOG_FILE, "a") as f:
            f.write(f"{os.path.basename(p)}: {repr(e)}\n")
        continue

if no_overlap:
    with open(os.path.join(OUT_ROOT, 'lesions_no_overlap.txt'), 'w') as f:
        f.write('\n'.join(no_overlap))

elapsed = time.time() - t0
print(f"Total time: {timedelta(seconds=int(elapsed))}")
print(f"[Summary] lesions={len(lesion_paths)}, subjects={len(gsp_1000_func_imgs)}, no-overlap={len(no_overlap)}")
