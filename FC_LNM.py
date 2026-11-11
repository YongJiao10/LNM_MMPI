import subprocess
import os, sys
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from nilearn.interfaces.fmriprep import load_confounds_strategy, load_confounds
from nilearn.maskers import NiftiSpheresMasker, NiftiMasker, NiftiLabelsMasker
from scipy.stats import zscore, pearsonr
from nilearn.connectome import ConnectivityMeasure

import multiprocessing
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

def roi_to_voxel_FC(subj, subj_func_path, roi_path, mask_path, fisherZ_output_path):
    assert os.path.exists(subj_func_path), f'Functional image: {subj_func_path} does not exist, please check again ~'
    assert os.path.exists(roi_path), f'ROI image: {roi_path} does not exist, please check again ~'
    assert os.path.exists(mask_path), f'mask image: {mask_path} doese not exist, please check again ~'
    
    print ('Make sure roi, mask and functional images are in the same space and orientation !')
    print(f'{subj} processing starts ...')
    
    roi_makser = NiftiLabelsMasker(
        labels_img=roi_path,
        standardize='zscore_sample',
        standardize_confounds='zscore_sample',t_r=3
    )
    seed_time_series = roi_makser.fit_transform(subj_func_path)
    
    brain_masker = NiftiMasker(mask_img=mask_path, standardize='zscore_sample', t_r=3)
    brain_time_series = brain_masker.fit_transform(subj_func_path)
    
    seed_to_voxel_correlations = (
        np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0]
    )
    
    # seed_to_voxel_correlations = np.clip(seed_to_voxel_correlations, -0.999999, 0.999999)
    # seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)

    seed_to_voxel_correlations_fisher_z = np.arctanh(seed_to_voxel_correlations)
    seed_to_voxel_correlations_fisher_z_img = brain_masker.inverse_transform(seed_to_voxel_correlations_fisher_z.T)
    
    seed_to_voxel_correlations_fisher_z_img.to_filename(fisherZ_output_path)
    # seed_to_voxel_correlations_img.to_filename(corr_output_path) 
    print(f'sub-{subj} finished whole brain FC ~ \n')

def avg_connectivity_calculation(source_dir, image_name, mask_image, T_threshold, P_threshold):
    """
    Given a bunch of functional/structural connectiviy Fisher Z maps , 
    calculate the mean avgerage Fisher-Z/corr maps and calculate the 1-sample-ttest T/P maps
    Arg:
        source_dir: Prepare the Z-maps in a this directory, images should be nifti format
        image_name: common image name like sub-*_FC.FisherZ.nii.gz
        target_dir: Output directory
        mask_image: Pathlike str or nilearn.Image
        T_threshold: the t threshold for 1-sample ttest
        P_threshold: the p threshold for 1-sample ttest
    
    Return:
        avgZmaps, avgRmaps, raw_T_maps, raw_P_maps, threshold_T_maps, threshold_P_maps
    """
    ## loading mask and making a binary data (contains True and False) for computation 
    # if mask_image is not None and is_pathlike(mask_image):
    #     assert os.path.exists(mask_image), f'{mask_image} does not exist, please check it again'
    #     mask_data = nib.load(mask_image).get_fdata().astype(bool) 
    # else:
    #     mask_data = mask_image.get_fdata().astype(bool)
    
    from scipy.stats import ttest_1samp
    from glob import glob
    print("+++ One lesion would cost several mins, have a break when the program is runnning :)")
    connectivity_maps = sorted(glob(f'{source_dir}/{image_name}'))
    mask_data = nib.load(mask_image).get_fdata().astype(bool)
    n_subjects = len(connectivity_maps)
    
    if len(nib.load(connectivity_maps[0]).get_fdata().shape) == 4:
        avgZmaps_data = np.average([np.nan_to_num(nib.load(i).get_fdata()) for i in connectivity_maps], axis=0)
        avgRmaps_data = np.tanh(avgZmaps_data)

        ## should be (n_subjects, x, y,z)
        connectivity_maps_data = np.array([np.nan_to_num(nib.load(i).get_fdata()[:,:,:,0]) for i in connectivity_maps])
        
    elif len(nib.load(connectivity_maps[0]).get_fdata().shape) == 3:
        avgZmaps_data = np.average([np.nan_to_num(nib.load(i).get_fdata()) for i in connectivity_maps], axis=0)
        avgRmaps_data = np.tanh(avgZmaps_data)

        ## should be (n_subjects, x, y,z)
        connectivity_maps_data = np.array([np.nan_to_num(nib.load(i).get_fdata()) for i in connectivity_maps])
    
    avgZmaps = nib.Nifti1Image(avgZmaps_data, affine=nib.load(connectivity_maps[0]).affine)
    avgRmaps = nib.Nifti1Image(avgRmaps_data, affine=nib.load(connectivity_maps[0]).affine)

    masked_connectivity_maps_data = connectivity_maps_data[:,mask_data]
    connectivity_maps_data_reshaped = masked_connectivity_maps_data.reshape(n_subjects, -1)
    
    ## 1 sample ttest
    t_values = []
    p_values = []
    ## this will cost a lot of time
    for i in range(connectivity_maps_data_reshaped.shape[1]):
        t_val, p_val = ttest_1samp(connectivity_maps_data_reshaped[:,i], popmean=0, axis=0)
        t_values.append(t_val)
        p_values.append(p_val)
    
    t_map_full = np.zeros(mask_data.shape)
    t_map_full[mask_data] = t_values

    p_map_full = np.zeros(mask_data.shape)
    p_map_full[mask_data] = p_values

    raw_T_maps = nib.Nifti1Image(t_map_full, affine=nib.load(connectivity_maps[0]).affine)
    raw_P_maps = nib.Nifti1Image(p_map_full, affine=nib.load(connectivity_maps[0]).affine)

    if T_threshold is None:
        threshold_T_maps = nib.Nifti1Image(t_map_full, affine=nib.load(connectivity_maps[0]).affine)
    else:
        threshold_T_map_data = np.where(np.abs(t_map_full) > T_threshold, t_map_full, 0)
        threshold_T_maps = nib.Nifti1Image(threshold_T_map_data, affine=nib.load(connectivity_maps[0]).affine)


    if P_threshold is None:
        threshold_P_maps = nib.Nifti1Image(p_map_full, affine=nib.load(connectivity_maps[0]).affine)
    else:
        threshold_P_map_data = np.where(np.abs(p_map_full) < P_threshold, p_map_full, 0)
        threshold_P_maps = nib.Nifti1Image(threshold_P_map_data, affine=nib.load(connectivity_maps[0]).affine)
    
    return avgZmaps, avgRmaps, raw_T_maps, raw_P_maps, threshold_T_maps, threshold_P_maps


def main():
    # mask_path_1mm = '/home/yojiao/LSS/jianglab/3_Data_Working/LNM/Data/mask_atlas/MNI152_T1_1mm_brain_mask.nii.gz'
    mask_path_2mm = '/home/yojiao/LSS/jianglab/3_Data_Working/masks/MNI152_T1_2mm_brain_mask.nii'

    gsp_1000_dataset = '/home/yojiao/LSS/jianglab/3_Data_Working/GSP1000_v2_dataset/'
    gsp_1000_func_imgs = glob(f'{gsp_1000_dataset}/*/func/sub-*_bld_skip4*.nii.gz')

    gsp_1000_sublist = []
    with open (f'{gsp_1000_dataset}/GSP1000.txt', 'r') as f:
        for line in f.readlines():
            gsp_1000_sublist.append(line.strip())


    grafman_metadata_df = pd.read_csv('/home/yojiao/LSS/jianglab/3_Data_Working/LNM/Data/metadata/demographicalData.cleaning.ziyan.250306.v2.csv')
    lesion_paths = [f'/home/yojiao/LSS/jianglab/3_Data_Working/LNM/Data/Lesion_files/{i.split("resampled_")[-1]}.nii' for i in grafman_metadata_df["Lesion_name"]]
    lesion_names = [f'{i.split("resampled_")[-1]}' for i in grafman_metadata_df["Lesion_name"]]
    
    print("---------------------------------")
    start_time = time.time()
    for lesion, lesion_subj in zip(lesion_paths[3:], lesion_names[3:]):
        print(f'++++ For {lesion_subj} -- Start FC processing ....')
        params = [(subj, subj_func_img, lesion, '/home/yojiao/LSS/jianglab/3_Data_Working/masks/MNI152_T1_2mm_brain_mask.nii', f'/home/yojiao/LSS/jianglab/3_Data_Working/LNM/Output/grafman_functional_LNM/{lesion_subj}/{subj}_FC.FisherZ.nii.gz') for subj, subj_func_img in zip(gsp_1000_sublist, sorted(gsp_1000_func_imgs))]
        if not os.path.exists(f'/home/yojiao/LSS/jianglab/3_Data_Working/LNM/Output/grafman_functional_LNM/{lesion_subj}'):
            os.makedirs(f'/home/yojiao/LSS/jianglab/3_Data_Working/LNM/Output/grafman_functional_LNM/{lesion_subj}')
            
        with multiprocessing.Pool(processes=20) as pool:
            pool.starmap(roi_to_voxel_FC, params)
        print(f'++++ For {lesion_subj} --- Processing finished !\n\n')
    
    end_time = time.time()
    running_time = (end_time - start_time) / 3600
    print(f'Processing time is {running_time} hours')
    print("---------------------------------")


    T_threshold = 10
    start_time = time.time()
    iowa_subList = [i.split('/')[-1] for i in sorted(glob('/home/yojiao/LSS/jianglab/3_Data_Working/LNM/Output/iowa_functional_LNM/sub-*'))]
    for subj in iowa_subList[2:]:
        subj_dir = f'/home/yojiao/LSS/jianglab/3_Data_Working/LNM/Output/iowa_functional_LNM/{subj}'
        subj_output_dir = f'/home/yojiao/LSS/jianglab/3_Data_Working/LNM/Output/iowa_fLNM_Summary/{subj}'
        print('--------------------------------------------')
        print(f'+++ {subj} Processing starts...')
        avgZmaps, avgRmaps, raw_T_maps, raw_P_maps, threshold_T_maps, threshold_P_maps = avg_connectivity_calculation(source_dir=subj_dir, image_name='*.FisherZ.nii.gz', 
                                                                                                                      mask_image=mask_path_2mm, 
                                                                                                                      T_threshold=T_threshold, 
                                                                                                                      P_threshold=None
                                                                                                                     )
        output_raw_Tmaps = f'{subj_output_dir}/{subj}_GSP1000_rawTmaps.nii.gz'
        output_raw_Pmaps = f'{subj_output_dir}/{subj}_GSP1000_rawPmaps.nii.gz'
        output_avgZmaps = f'{subj_output_dir}/{subj}_GSP1000_avgZmaps.nii.gz'
        output_avgRmaps = f'{subj_output_dir}/{subj}_GSP1000_avgRmaps.nii.gz'
        thrs_T_maps = f'{subj_output_dir}/{subj}_thrs{T_threshold}_Tmaps.nii.gz'
    
        avgZmaps.to_filename(output_avgZmaps)
        avgRmaps.to_filename(output_avgRmaps)
        raw_T_maps.to_filename(output_raw_Tmaps)
        raw_P_maps.to_filename(output_raw_Pmaps)
        threshold_T_maps.to_filename(thrs_T_maps)
        
        print(f'+++ {subj} Processing finished~')
    
        print('--------------------------------------------\n')
    
    end_time = time.time()
    running_time = (end_time - start_time) / 3600
    print(f'Running time: {running_time:.3f} hrs')

if __name__ == '__main__':

    ROOT = '/home/yojiao/'
    main()
    
    ## Running
    
    
    
    
        