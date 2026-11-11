# === Fast FC with subject-outer loop (clean) ===
import os, glob, time
import numpy as np
import nibabel as nib
from tqdm import tqdm
from nilearn import image
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
from datetime import timedelta

# ---- Paths ----
# atlas = '/home/yojiao/LSS/jianglab/3_Data_Working/masks/BN_Atlas_246_2mm.nii.gz'
atlas = '/home/yojiao/sfc_mmpi/atlas/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_MNI152NLin6Asym_2mm.nii.gz'
BRAIN_MASK = '/home/yojiao/LSS/jianglab/3_Data_Working/masks/MNI152_T1_2mm_brain_mask.nii'
LESION_DIR = '/home/yojiao/LSS/boeslab/Data/Lesion/Iowa/Registry/lesionMasks/Chronic/MNI152/2mm'
OUT_ROOT = '/home/yojiao/hpchome/GSP_FC_Lesion_Iowa'
os.makedirs(OUT_ROOT, exist_ok=True)

# ---- GSP ----
gsp_1000_dataset = '/home/yojiao/LSS/jianglab/3_Data_Working/GSP1000_v2_dataset/'
gsp_1000_func_imgs = sorted(glob.glob(f'{gsp_1000_dataset}/*/func/sub-*_bld_skip4*.nii.gz'))

# ---- Load atlas & brain mask ----
bn_img = nib.load(atlas)
bn_data = bn_img.get_fdata()
brain_img = nib.load(BRAIN_MASK)

# ---- Corr measure ----
corr = ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)

# ---- Lesions ----
lesion_paths = sorted(glob.glob(f'{LESION_DIR}/*.nii*'))
no_overlap = []

t0 = time.time()
for p in tqdm(lesion_paths, desc='lesions'):
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
    func_imgs_nifti = [nib.load(pth) for pth in gsp_1000_func_imgs]
    sigs = [masker.transform(img) for img in func_imgs_nifti]

    feats = corr.fit_transform(sigs)
    feats = np.arctanh(feats)
    np.save(os.path.join(OUT_ROOT, f'{lesion_sub}.npy'), feats)

if no_overlap:
    with open(os.path.join(OUT_ROOT, 'lesions_no_overlap_in_BN246.txt'), 'w') as f:
        f.write('\n'.join(no_overlap))

elapsed = time.time() - t0
print(f"Total time: {timedelta(seconds=int(elapsed))}")
