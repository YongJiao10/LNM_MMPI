import numpy as np
import pandas as pd
import torch, random, os, sys, math, scipy, copy, re
from argparse import ArgumentParser
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, norm, ttest_ind, mannwhitneyu, chi2_contingency, kruskal, fisher_exact, ttest_1samp
from statsmodels.stats.multitest import multipletests, fdrcorrection
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture as GMM
import statsmodels.api as sm
from sklearn.metrics import r2_score

from typing import Dict, List, Optional, Literal, Tuple
from collections import OrderedDict, defaultdict
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pingouin as pg

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = '/Users/jy/Documents/work/BrainData/'
# R_ROOT = r'C:\Program Files\R\R-4.4.2' 
ABCD_CORE = os.path.join(ROOT, 'ABCD/TabulatedData_BehaviorData_Release5.1/core/')


SEED = 0

SUBTYPE_NAME = ['Externalizing-Disinhibited', 'Internalizing-Distressed', 'Mixed Dysregulation']
SUBTYPE_ABBR = ['ED', 'ID', 'MD']

# NIH Toolbox Cognition Battery
NIH_KEY = ['nihtbx_flanker_uncorrected', 'nihtbx_list_uncorrected', 'nihtbx_cardsort_uncorrected', 'nihtbx_pattern_uncorrected', 
            'nihtbx_picvocab_uncorrected', 'nihtbx_picture_uncorrected', 'nihtbx_reading_uncorrected', 'nihtbx_fluidcomp_uncorrected', 'nihtbx_cryst_uncorrected', 'nihtbx_totalcomp_uncorrected']
NIH_KEY_AGE = ['nihtbx_flanker_agecorrected', 'nihtbx_list_agecorrected', 'nihtbx_cardsort_agecorrected', 'nihtbx_pattern_agecorrected',
                'nihtbx_picvocab_agecorrected', 'nihtbx_picture_agecorrected', 'nihtbx_reading_agecorrected', 'nihtbx_fluidcomp_agecorrected', 'nihtbx_cryst_agecorrected', 'nihtbx_totalcomp_agecorrected']
NIH_NAME = ['Attention(NIH)', 'Working Memory(NIH)', 'Executive Function(NIH)', 'Processing Speed(NIH)',
            'Vocabulary(NIH)', 'Episodic Memory(NIH)', 'Reading(NIH)', 'Fluid Cognition(NIH)', 'Crystallized Cognition(NIH)', 'Overall Cognition(NIH)']

# WISC-V
WISC_KEY = ['pea_wiscv_trs']
WISC_KEY_S = ['pea_wiscv_tss']
WISC_NAME = ['Fluid Intelligence(WISC-V)']
# CBCL
CBCL_KEY = ['cbcl_scr_syn_anxdep_r', 'cbcl_scr_syn_withdep_r', 'cbcl_scr_syn_somatic_r', 'cbcl_scr_syn_social_r', 
              'cbcl_scr_syn_thought_r', 'cbcl_scr_syn_attention_r', 'cbcl_scr_syn_rulebreak_r', 'cbcl_scr_syn_aggressive_r', 
            'cbcl_scr_syn_internal_r', 'cbcl_scr_syn_external_r', 'cbcl_scr_syn_totprob_r']
CBCL_KEY_T = ['cbcl_scr_syn_anxdep_t', 'cbcl_scr_syn_withdep_t', 'cbcl_scr_syn_somatic_t', 'cbcl_scr_syn_social_t', 
              'cbcl_scr_syn_thought_t', 'cbcl_scr_syn_attention_t', 'cbcl_scr_syn_rulebreak_t', 'cbcl_scr_syn_aggressive_t', 
              'cbcl_scr_syn_internal_t', 'cbcl_scr_syn_external_t', 'cbcl_scr_syn_totprob_t']
CBCL_NAME = ['Anxious Depressed(CBCL)', 'Withdrawn Depressed(CBCL)', 'Somatic Complaints(CBCL)', 'Social Problems(CBCL)', 
             'Thought Problems(CBCL)', 'Attention Problems(CBCL)', 'Rule-Breaking Behavior(CBCL)', 'Aggressive Behavior(CBCL)', 
            'Internalizing Problems(CBCL)', 'Externalizing Problems(CBCL)', 'Total Problems(CBCL)']
CBCL_NAME_ = ['Anxious Depressed', 'Withdrawn Depressed', 'Somatic Complaints', 'Social Problems', 
             'Thought Problems', 'Attention Problems', 'Rule-Breaking Behavior', 'Aggressive Behavior', 
            'Internalizing Problems', 'Externalizing Problems', 'Total Problems']

CBCL_DSM = ['Anxiety Disorder', 'ADHD', 'Depression', 'Conduct Disorder', 'Somatic Problems', 'Oppositional Defiant Disorder']
CBCL_DSM_KEY = ['cbcl_scr_dsm5_anxdisord_t', 'cbcl_scr_dsm5_adhd_t', 'cbcl_scr_dsm5_depress_t', 'cbcl_scr_dsm5_conduct_t', 'cbcl_scr_dsm5_somaticpr_t', 'cbcl_scr_dsm5_opposit_t']

SCALE_KEY = NIH_KEY[:4] + WISC_KEY + CBCL_KEY[:8]
SCALE_KEY_CORR = NIH_KEY_AGE[:4] + WISC_KEY_S + CBCL_KEY_T[:8]
# SCALES = ['Attention(NIH)', 'Working Memory(NIH)', 'Executive Function(NIH)', 'Processing Speed(NIH)', 'Fluid Intelligence(WISC-V)',
#           'Anxious Depressed(CBCL)', 'Withdrawn Depressed(CBCL)', 'Somatic Complaints(CBCL)', 'Social Problems(CBCL)', 
#           'Thought Problems(CBCL)', 'Attention Problems(CBCL)', 'Rule-Breaking Behavior(CBCL)', 'Aggressive Behavior(CBCL)']

SCALES = [
    # "Emotional/Internalizing Dysfunction (EID)",
    # "Thought Dysfunction (THD)",
    # "Behavioral/Externalizing Dysfunction (BXD)",

    # "Demoralization (RCd)",
    # "Somatic Complaints (RC1)",
    # "Low Positive Emotions (RC2)",
    # "Cynicism (RC3)",
    # "Antisocial Behavior (RC4)",
    # "Ideas of Persecution (RC6)",
    # "Dysfunctional Negative Emotions (RC7)",
    # "Aberrant Experiences (RC8)",
    # "Hypomanic Activation (RC9)",

    # "Aggressiveness-Revised (AGGR-r)",
    # "Psychoticism-Revised (PSYC-r)",
    # "Disconstraint-Revised (DISC-r)",
    # "Negative Emotionality/Neuroticism (NEGE-r)",
    # "Introversion/Low Positive Emotionality (INTR-r)",

    # Somatic Scales
    "Malaise (MLS)",
    "Gastro-Intestinal Complaints (GIC)",
    "Head Pain Complaints (HPC)",
    "Neurological Complaints (NUC)",
    "Cognitive Complaints (COG)",

    # Internalizing Scales
    "Suicidal/Death Ideation (SUI)",
    "Helplessness/Hopelessness (HLP)",
    "Self-Doubt (SFD)",
    "Inefficacy (NFC)",
    "Stress/Worry (STW)",
    "Anxiety (AXY)",
    "Anger Proneness (ANP)",
    "Behavior-Restricting Fears (BRF)",
    "Multiple Specific Fears (MSF)",

    # Externalizing Scales
    "Juvenile Conduct Problems (JCP)",
    "Substance Abuse (SUB)",
    "Aggression (AGG)",
    "Activation (ACT)",

    # Interpersonal Scales
    "Family Problems (FML)",
    "Interpersonal Passivity (IPP)",
    "Social Avoidance (SAV)",
    "Shyness (SHY)",
    "Disaffiliativeness (DSF)",

    # Interest
    "Aesthetic-Literary Interests (AES)",
    "Mechanical-Physical Interests (MEC)",

    # PSY-5
    "Aggressiveness-Revised (AGGR-r)",
    "Psychoticism-Revised (PSYC-r)",
    "Disconstraint-Revised (DISC-r)",
    "Negative Emotionality/Neuroticism (NEGE-r)",
    "Introversion/Low Positive Emotionality (INTR-r)"
]

# ---------- MMPI display label control ----------
MMPI_LABEL_STYLE = "full_no_abbr"
MMPI_LABEL_STYLES = ("full_no_abbr", "full_with_abbr")

MMPI_SCALE_KEYS_23 = [
    "tMLS", "tGIC", "tHPC", "tNUC", "tCOG",
    "tSUI", "tHLP", "tSFD", "tNFC", "tSTW", "tAXY", "tANP", "tBRF", "tMSF",
    "tJCP", "tSUB", "tAGG", "tACT",
    "tFML", "tIPP", "tSAV", "tSHY", "tDSF",
]

MMPI_LABELS_WITH_ABBR_23 = [
    "Malaise (MLS)",
    "Gastro-Intestinal Complaints (GIC)",
    "Head Pain Complaints (HPC)",
    "Neurological Complaints (NUC)",
    "Cognitive Complaints (COG)",
    "Suicidal/Death Ideation (SUI)",
    "Helplessness/Hopelessness (HLP)",
    "Self-Doubt (SFD)",
    "Inefficacy (NFC)",
    "Stress/Worry (STW)",
    "Anxiety (AXY)",
    "Anger Proneness (ANP)",
    "Behavior-Restricting Fears (BRF)",
    "Multiple Specific Fears (MSF)",
    "Juvenile Conduct Problems (JCP)",
    "Substance Abuse (SUB)",
    "Aggression (AGG)",
    "Activation (ACT)",
    "Family Problems (FML)",
    "Interpersonal Passivity (IPP)",
    "Social Avoidance (SAV)",
    "Shyness (SHY)",
    "Disaffiliativeness (DSF)",
]

MMPI_STYLE_ALIASES = {
    "no_abbr": "full_no_abbr",
    "with_abbr": "full_with_abbr",
}


def mmpi_strip_abbr(label: str) -> str:
    """Strip a trailing parenthetical abbreviation from a scale label."""
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", str(label)).strip()
    return stripped if stripped else str(label)


def _resolve_mmpi_label_style(style: Optional[str]) -> str:
    style = MMPI_LABEL_STYLE if style is None else style
    style = MMPI_STYLE_ALIASES.get(style, style)
    if style not in MMPI_LABEL_STYLES:
        raise ValueError(
            f"Unsupported MMPI label style: {style}. "
            f"Supported styles: {MMPI_LABEL_STYLES} and aliases: {tuple(MMPI_STYLE_ALIASES.keys())}"
        )
    return style


def get_mmpi_label_map(keys=None, style: Optional[str] = None) -> Dict[str, str]:
    """
    Return key -> display label mapping for MMPI scales.
    Keeps input order and skips keys outside canonical MMPI_SCALE_KEYS_23.
    """
    style = _resolve_mmpi_label_style(style)
    key_to_with_abbr = dict(zip(MMPI_SCALE_KEYS_23, MMPI_LABELS_WITH_ABBR_23))
    key_to_no_abbr = {k: mmpi_strip_abbr(v) for k, v in key_to_with_abbr.items()}
    key_to_label = key_to_no_abbr if style == "full_no_abbr" else key_to_with_abbr

    ordered_keys = MMPI_SCALE_KEYS_23 if keys is None else list(keys)
    label_map = OrderedDict()
    for k in ordered_keys:
        if k in key_to_label:
            label_map[k] = key_to_label[k]
    return dict(label_map)


def get_mmpi_labels(keys=None, style: Optional[str] = None) -> List[str]:
    """Return MMPI display labels in the same order as valid input keys."""
    return list(get_mmpi_label_map(keys=keys, style=style).values())


def Connectome(sigs, kind='correlation', r2z=True, demean=True):
    # sigs: list of time series of subject with shape of (timepoints, num_ROIs)
    corr_measure = ConnectivityMeasure(
                                       kind=kind,
                                       vectorize=True,
                                       discard_diagonal=True)

    features = corr_measure.fit_transform(sigs)
    if r2z:
        features = np.arctanh(features)
    if demean:
        features = features - features.mean(axis=1, keepdims=True)

    return features.squeeze().astype(np.float32)

def seed_all(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



ICC_METHOD = "foldwise_unit_l2_ICC1"


def compute_icc(weights: np.ndarray) -> float:
    """Compare aligned loading patterns (folds × features), ignoring norm.

    Each complete view is normalized separately in every fold. Combined brain
    vectors include all 232 features. The one-way ANOVA expression is identical
    to Pingouin's single-rater absolute-agreement ICC1.
    """
    patterns = np.asarray(weights, dtype=float)
    patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)
    if np.all(patterns.var(axis=0, ddof=1) == 0):
        return 1.0
    k, n = patterns.shape
    means = patterns.mean(axis=0)
    between = k * means.var(ddof=1)
    within = np.sum((patterns - means) ** 2) / (n * (k - 1))
    return round(float((between - within) / (between + (k - 1) * within)), 4)


def unique_in_order(arr):
    '''
    Return unique elements in the order of their first occurrence.
    '''
    _, idx = np.unique(arr, return_index=True)
    sorted_idx = np.sort(idx)
    return arr[sorted_idx], sorted_idx

def filter_baseline(df):
    return df[df['eventname'] == 'baseline_year_1_arm_1'].drop('eventname', axis=1)

def filter_year(df, year):
    if year == 0:
        eventname = 'baseline_year_1_arm_1'
    else:
        eventname = f'{year}_year_follow_up_y_arm_1'
    return df[df['eventname'] == eventname].drop('eventname', axis=1)

def row_has_same_values(array):
    # exclude bad data
    return np.any(np.all(array[:,1:] == array[:,:-1], axis=1))

# def GCCA_align(U_behav_ref, U_behav_cv, U_fc_cv, fc_score, behav_score):
#     """
#     Align GCCA components and signs based on reference results.
    
#     Args:
#         U_behav_ref: Reference symptom loadings, shape (n_samples, n_dims)
#         U_behav_cv: CV symptom loadings, shape (n_folds=10, n_samples, n_dims=13)
#         U_fc_cv: CV FC loadings, shape (n_folds=10, n_features=32, n_dims=13)
#         fc_score: List of arrays, each with shape (n_samples_fold_i, n_dims)
#         behav_score: List of arrays, each with shape (n_samples_fold_i, n_dims)
#     """
#     n_folds, _, n_dims = U_behav_cv.shape
    
#     U_behav_cv_aligned = U_behav_cv.copy()
#     U_fc_cv_aligned = U_fc_cv.copy()
#     fc_score_aligned = [score.copy() for score in fc_score]
#     behav_score_aligned = [score.copy() for score in behav_score]
    
#     for fold in range(n_folds):
#         df_combined = pd.DataFrame(
#             np.column_stack([U_behav_ref, U_behav_cv[fold]]),
#             columns=[f'ref_{i}' for i in range(n_dims)] + [f'cv_{i}' for i in range(n_dims)]
#         )
#         # Calculate correlation matrix between reference and CV loadings
#         corr_full = df_combined.corr()
#         corr_matrix = corr_full.iloc[:n_dims, n_dims:].values
        
#         # Find optimal dimension matching using Hungarian algorithm
#         _, dim_idx = linear_sum_assignment(-np.abs(corr_matrix))
        
#         # Adjust signs based on correlation with reference
#         signs = np.sign([corr_matrix[i, dim_idx[i]] for i in range(n_dims)])
        
#         # Apply signs to loadings
#         for arr in [U_behav_cv_aligned[fold], U_fc_cv_aligned[fold]]:
#             arr[:] = arr[:, dim_idx] * signs[None, :]
        
#         for arr in [fc_score_aligned[fold], behav_score_aligned[fold]]:
#             arr[:] = arr[:, dim_idx] * signs[None, :]
    
#     fc_score_aligned = np.concatenate(fc_score_aligned, axis=0)
#     behav_score_aligned = np.concatenate(behav_score_aligned, axis=0)
#     return U_behav_cv_aligned, U_fc_cv_aligned, fc_score_aligned, behav_score_aligned



    
def get_var_names(lst):
    global_vars = sys._getframe(1).f_globals
    result = []
    for item in lst:
        for var_name, var_value in global_vars.items():
            if var_value is item:
                result.append(var_name)
                break
    return result

def FC_vec2mat(x, n=100, upper=False, include_diag=False):
    k = 0 if include_diag else (1 if upper else -1)

    if upper:
        indices = np.triu_indices(n, k=k)
    else:
        indices = np.tril_indices(n, k=k)
    
    if x.ndim == 1:
        fc = np.zeros((n, n))
        fc[indices] = x
        fc.T[indices] = x
    elif x.ndim == 2:
        p, _ = x.shape
        fc = np.zeros((p, n, n))
        fc[:, indices[0], indices[1]] = x
        fc[:, indices[1], indices[0]] = x
    elif x.ndim == 3:
        p, q, _ = x.shape
        fc = np.zeros((p, q, n, n))
        fc[:, :, indices[0], indices[1]] = x
        fc[:, :, indices[1], indices[0]] = x
    
    return fc.astype(np.float32)

def FC_mat2vec(fc, upper=False, include_diag=False):
    n = fc.shape[-1]

    k = 0 if include_diag else (1 if upper else -1)
    
    if upper:
        indices = np.triu_indices(n, k=k)
    else:
        indices = np.tril_indices(n, k=k)
    
    if fc.ndim == 2:
        x = fc[indices]
    elif fc.ndim == 3:
        x = fc[:, indices[0], indices[1]]
    elif fc.ndim == 4:
        x = fc[:, :, indices[0], indices[1]]
    else:
        raise ValueError("Input array must have 2, 3, or 4 dimensions")
    
    return x.astype(np.float32)

def calculate_network_fc(FC):
    """
    Calculate average functional connectivity between networks.
    
    Parameters:
    -----------
    FC : numpy.ndarray
        Input functional connectivity matrices with shape (N, 100, 100) 
        where N is the number of subjects.
    
    Returns:
    --------
    numpy.ndarray
        Average connectivity between networks with shape (N, 7, 7).
    """
    if FC.shape[-1] != 100:
        FC = FC_vec2mat(FC)
        
    # Define network indices
    networks = [
        np.concatenate([np.arange(9), np.arange(50, 58)]),          # VN
        np.concatenate([np.arange(9, 15), np.arange(58, 66)]),      # SMN
        np.concatenate([np.arange(15, 23), np.arange(66, 73)]),     # DAN
        np.concatenate([np.arange(23, 30), np.arange(73, 78)]),     # VAN
        np.concatenate([np.arange(30, 33), np.arange(78, 80)]),     # LN
        np.concatenate([np.arange(33, 37), np.arange(80, 89)]),     # FPCN
        np.concatenate([np.arange(37, 50), np.arange(89, 100)])     # DMN
    ]
    
    n_subjects = FC.shape[0]
    connectome = np.zeros((n_subjects, 7, 7))
    
    # Compute all network pairs at once for each subject
    for n in range(n_subjects):
        for i, net_i in enumerate(networks):
            for j, net_j in enumerate(networks):
                connectome[n, i, j] = FC[n][net_i][:, net_j].mean()
                
    return FC_mat2vec(connectome, include_diag=True)

def get_data_splits(IDs, kf):
    splits = []
    unique_ids, indices = np.unique(IDs, return_inverse=True)
    for train_index, test_index in kf.split(unique_ids):
        train_mask = np.isin(indices, train_index)
        test_mask = np.isin(indices, test_index)
        original_train_index = np.where(train_mask)[0]
        original_test_index = np.where(test_mask)[0]
        splits.append((original_train_index, original_test_index))
    return splits

def get_stratified_data_splits(IDs, labels, n_splits, random_state=42):
    from sklearn.model_selection import StratifiedKFold
    splits = []
    # 获取唯一ID和对应的索引映射
    unique_ids, indices = np.unique(IDs, return_inverse=True)
    
    # 为每个唯一ID分配对应的标签
    # 如果一个ID对应多个样本，取第一个样本的标签
    unique_labels = []
    for id_idx in range(len(unique_ids)):
        mask = indices == id_idx
        unique_labels.append(labels[np.where(mask)[0][0]])
    unique_labels = np.array(unique_labels)
    
    # 使用StratifiedKFold进行分层采样
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 对唯一ID进行分层折叠
    for train_index, test_index in skf.split(unique_ids, unique_labels):
        # 将唯一ID的划分映射回原始数据的索引
        train_mask = np.isin(indices, train_index)
        test_mask = np.isin(indices, test_index)
        original_train_index = np.where(train_mask)[0]
        original_test_index = np.where(test_mask)[0]
        splits.append((original_train_index, original_test_index))
        
    return splits
    
    
def rearrange_FC(mat):
    """
    重新排列相关性矩阵，使同一网络的左右部分合并在一起。
    """
    # 计算每个网络的左右部分的索引
    pos = [9, 15, 23, 30, 33, 37, 50, 58, 66, 73, 78, 80, 89]
    left_indices = []
    right_indices = []
    
    for i in range(7):
        if i == 0:
            left_indices.append(list(range(0, pos[i])))
        else:
            left_indices.append(list(range(pos[i-1], pos[i])))
    
    for i in range(7, 14):
        if i == 13:
            right_indices.append(list(range(pos[i-1], 100)))
        else:
            right_indices.append(list(range(pos[i-1], pos[i])))

    # 打印检查生成的索引
    # print("Left indices:", left_indices)
    # print("Right indices:", right_indices)

    # 生成新的排序顺序
    new_order = []
    new_pos = []
    current_index = 0
    for left, right in zip(left_indices, right_indices):
        new_order.extend(left)
        new_order.extend(right)
        current_index += len(left) + len(right)
        new_pos.append(current_index)

    # 去掉最后一个多余的分割位置
    new_pos.pop()

    # 打印检查新顺序
    # print("New order:", new_order)
    # print("New pos:", new_pos)

    # 重新排列矩阵
    if mat.ndim == 3:
        new_mat = mat[:, new_order, :][:, :, new_order]
    else:
        new_mat = mat[np.ix_(new_order, new_order)]

    return new_mat, new_pos


def calculate_angles(loadings):
    n = loadings.shape[1]
    angles = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dot_product = np.dot(loadings[:, i], loadings[:, j])
            norm_i = np.linalg.norm(loadings[:, i])
            norm_j = np.linalg.norm(loadings[:, j])
            cos_angle = dot_product / (norm_i * norm_j)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles[i, j] = angles[j, i] = np.degrees(angle)
    return angles

def average_by_id(ids, scores):
    """
    Calculate the mean of scores for each unique ID.

    Parameters:
    ids (list): List of IDs
    scores: Either a single score array/values, or a list of score arrays/values,
           where the first dimension (or the value itself) corresponds to IDs

    Returns:
    If input is single score array/values: array with averaged scores
    If input is list of score arrays/values: list of arrays with averaged scores
    """
    # Convert single score input to list for unified processing
    single_input = not isinstance(scores, list)
    scores_list = [scores] if single_input else scores
    
    # Create a dictionary to store indices for each ID
    id_indices = defaultdict(list)
    for i, id in enumerate(ids):
        id_indices[id].append(i)

    # Preserve the original order of unique IDs
    unique_ids = []
    seen = set()
    for id in ids:
        if id not in seen:
            unique_ids.append(id)
            seen.add(id)

    # Calculate the average score for each input array/value
    result = []
    for scores in scores_list:
        scores_array = np.array(scores)
        
        # Check if scores is 1D (scalar values) or multi-dimensional
        if scores_array.ndim == 1:
            # Handle scalar values
            avg_scores = np.array([np.mean(scores_array[id_indices[id]]) for id in unique_ids])
        else:
            # Handle vector values
            # Ensure the first dimension matches the length of IDs
            assert scores_array.shape[0] == len(ids), "The first dimension must match the length of IDs"
            avg_scores = np.array([np.mean(scores_array[id_indices[id]], axis=0) for id in unique_ids])
            
        result.append(avg_scores)

    # Return single result if input was single score
    return result[0] if single_input else result


def rearrange_cluster_labels(labels, probs, label_mapping=None):
    """Rearrange cluster labels and probability matrix columns based on ascending order of run label frequencies."""
    from collections import Counter
    
    return_mapping = False
    if label_mapping is None:
        return_mapping = True
        # Count the frequency of each label
        label_counts = Counter(labels)
        
        # Sort labels by frequency, smallest group first 
        sorted_labels = sorted(label_counts, key=label_counts.get)
        
        # Create mapping from original label to new label
        label_mapping = {label: i for i, label in enumerate(sorted_labels)}
    
    # Apply the mapping to rearrange labels
    rearranged_labels = np.array([label_mapping[label] for label in labels])
    
    # Get the inverse mapping to rearrange probs columns
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Create a list of column indices in the order we want
    col_order = [inverse_mapping[i] for i in range(len(label_mapping))]
    
    # Reorder probs using the column order
    rearranged_probs = probs[:, col_order]
    
    if return_mapping:
        return rearranged_labels, rearranged_probs, label_mapping
    return rearranged_labels, rearranged_probs


def clustering_acc_and_align(Y, Y_pred, Y_pred_run=None, Y_pred_prob=None):
    
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    # Compute the pairwise count using numpy
    np.add.at(w, (Y_pred, Y), 1)
    
    # Use the Hungarian algorithm to find the best matching
    row_ind, col_ind = linear_sum_assignment(-w)
    
    # Calculate accuracy
    accuracy = w[row_ind, col_ind].sum() / Y_pred.size
    
    # Create a mapping dictionary
    mapping = dict(zip(row_ind, col_ind))
    
    # Apply the mapping to the predicted labels
    aligned_labels = np.array([mapping[label] for label in Y_pred])
    
    # Apply mapping to Y_pred_run if provided
    aligned_labels_run = None
    if Y_pred_run is not None:
        aligned_labels_run = np.array([mapping[label] for label in Y_pred_run])
    
    # Reorder Y_pred_prob if provided
    aligned_probs = None
    if Y_pred_prob is not None:
        aligned_probs = np.zeros_like(Y_pred_prob)
        for original, mapped in mapping.items():
            aligned_probs[:, mapped] = Y_pred_prob[:, original]
    
    return accuracy, aligned_labels, aligned_labels_run, aligned_probs


def biotype_prevalence_similarity(p_counts, q_counts):
    p = np.array(p_counts) / np.sum(p_counts)
    q = np.array(q_counts) / np.sum(q_counts)
    jsd = jensenshannon(p, q, base=2)
    return 1 - jsd

def cohens_d(data1, data2):
    """Calculate Cohen's d for two groups
    
    Parameters:
    data1, data2: arrays of data for two groups
    
    Returns:
    d: Cohen's d effect size
    """
    n1, n2 = len(data1), len(data2)
    if n1 < 2 or n2 < 2:  # Require at least 2 values in each group
        return np.nan
        
    # Calculate means
    m1, m2 = np.mean(data1), np.mean(data2)
    
    # Calculate pooled standard deviation
    s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)  # ddof=1 for sample variance
    pooled_se = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    if pooled_se == 0:  # Avoid division by zero
        return np.nan
    
    return (m2 - m1) / pooled_se
    
def ttest_significance(fc1, fc2, alpha=0.05, fdr=True, return_d=False):
    # Ensure inputs are NumPy arrays
    fc1, fc2 = np.asarray(fc1), np.asarray(fc2)
    
    # Check if inputs are 2D arrays
    if fc1.ndim != 2 or fc2.ndim != 2:
        raise ValueError("Both groups must be 2D arrays.")
    
    # Check if number of features matches
    if fc1.shape[1] != fc2.shape[1]:
        raise ValueError("Both groups must have the same number of features.")
    
    # Perform t-test (Welch's t-test by default)
    t_stats, p_values = ttest_ind(fc1, fc2, axis=0, equal_var=False)
    
    # Calculate effect size (Cohen's d)
    mean_diff = np.mean(fc1, axis=0) - np.mean(fc2, axis=0)
    n1, n2 = fc1.shape[0], fc2.shape[0]
    var1, var2 = np.var(fc1, axis=0, ddof=1), np.var(fc2, axis=0, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Handle division by zero in Cohen's d calculation
    pooled_std[pooled_std == 0] = np.nan
    effect_sizes = mean_diff / pooled_std
    
    # Apply multiple testing correction
    if fdr:
        _, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    else:
        corrected_p_values = p_values
    
    # Determine significant features
    significant_mask = corrected_p_values < alpha
    
    # Set non-significant t-stats to zero
    t_stats[~significant_mask] = 0
    effect_sizes[~significant_mask] = 0
    if return_d:
        return t_stats, effect_sizes
    return t_stats

def fisher_ztest(r1, r2, n1, n2):
    
    # Fisher Z-transformation
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)

    # Standard error
    se_diff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))

    # Z-score for the difference
    z_diff = (z1 - z2) / se_diff

    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_diff)))

    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = 'N.S.'
    
    # print(f'p-value: {p_value:.4f}, Significance: {significance}')
    
    return p_value, significance

def fdr_select_significant_weights(w_cv, alpha=0.05, fdr=True):
    """
    Perform max-abs normalization + one-sample t-test across CV folds
    with optional FDR correction to identify statistically significant sparse loadings.

    Parameters:
    - w_cv: numpy array of shape (n_folds, 4950, 2), each slice is a fold's loading
    - alpha: significance level (default 0.05)
    - fdr: whether to perform FDR correction (default True)

    Returns:
    - w_sig: array of shape (4950, 2), averaged significant weights (others are zero)
    - sig_count: array of shape (2,), number of significant features per dimension
    """
    # Normalize each fold and dimension by max absolute value
    for i in range(w_cv.shape[0]):
        for dim in range(2):
            max_abs = np.max(np.abs(w_cv[i, :, dim]))
            if max_abs > 0:
                w_cv[i, :, dim] /= max_abs

    w_sig = np.zeros((4950, 2))
    sig_count = []

    for dim in range(2):
        # Identify features that are non-zero in at least one fold
        mask_nonzero = np.any(w_cv[:, :, dim] != 0, axis=0)
        idx = np.where(mask_nonzero)[0]

        if len(idx) == 0:
            sig_count.append(0)
            continue

        # t-test on selected features
        t_vals, p_vals = ttest_1samp(w_cv[:, idx, dim], popmean=0, axis=0, nan_policy='omit')

        # Apply FDR correction or not
        if fdr:
            reject, _ = fdrcorrection(p_vals, alpha=alpha)
        else:
            reject = p_vals < alpha

        # Mean across folds only for significant features
        w_mean = w_cv[:, idx, dim].mean(axis=0)
        w_sig[idx[reject], dim] = w_mean[reject]
        sig_count.append(np.sum(reject))

    return w_sig, np.array(sig_count)

def add_demo_info(ids, labels, df, demo, lt, covariates=['sex', 'parental_education', 'income', 'marital']):
    
    ids_ = filter_baseline(df).index.intersection(demo.index).intersection(filter_baseline(lt).index).intersection(ids)
    idx = pd.Index(ids).get_indexer(ids_)
    labels = labels[idx]
    # Create a DataFrame for labels with 'src_subject_id' and 'Subtype'
    labels_df = pd.DataFrame({
        'src_subject_id': ids_, 
        'Subtype': labels
    })
    
    demo = demo.loc[ids_]
    lt = lt.loc[ids_]
    # Reset index of main DataFrame and demographic DataFrame to ensure src_subject_id is accessible
    df = df.loc[ids_]
    df = df.reset_index()

    # Merge subtype labels with the main DataFrame
    df = df.merge(labels_df, on='src_subject_id', how='left')
    
    # Select relevant demographic columns and merge them with the main DataFrame
    df = df.merge(demo.reset_index()[['src_subject_id'] + covariates], on='src_subject_id', how='left')
    # Merge interview age from lt using both 'src_subject_id' and 'eventname' as keys
    df = df.merge(lt.reset_index()[['src_subject_id', 'eventname', 'interview_age']], 
                  on=['src_subject_id', 'eventname'], how='left')

    # Store interview_age information in 'Time' column and drop the intermediate column
    df['Age'] = df.pop('interview_age')

    TIME_MAPPING = {
        'baseline_year_1_arm_1': 0,
        '1_year_follow_up_y_arm_1': 1,
        '2_year_follow_up_y_arm_1': 2,
        '3_year_follow_up_y_arm_1': 3,
        '4_year_follow_up_y_arm_1': 4
    }
    
    # Add time points based on event names
    df['Time'] = df['eventname'].map(TIME_MAPPING)

    # Format columns to appropriate types and rename for consistency
    df['Subtype'] = df['Subtype'].astype('string')
    df['Sex'] = df['sex']
    df['Marital'] = df['marital']
    df['Parental_Education'] = df['parental_education']
    df['Income'] = df['income']
    df.drop(columns=['sex', 'parental_education', 'income', 'marital'], inplace=True)
    return df

def LME_analysis(df, effects=['Subtype', 'Sex', 'Parental_Education', 'Income']):
    """
    Perform Linear Mixed Effects analysis on behavioral data with dynamic effects
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the behavioral data
    effects : list
        List of effects to include in the analysis. Each effect will be interacted with Time.
        Default: ['Subtype', 'Sex', 'Parental_Education', 'Income']
        
    Returns:
    --------
    tuple(pandas.DataFrame, pandas.DataFrame)
        - DataFrame with model predictions
        - DataFrame containing ANOVA results with FDR-corrected p-values and partial eta squared values
    """
    # os.environ['R_HOME'] = R_ROOT 
    from pymer4.models import Lmer
    # from rpy2.robjects import pandas2ri
    # pandas2ri.activate()
    
    # Step 1: Get unique subjects and define specified time points
    unique_subjects = df['src_subject_id'].unique()
    unique_ages = np.arange(9, 16) * 12
    
    # Step 2: Create a full data table with all combinations of subjects and time points
    all_combinations = pd.MultiIndex.from_product([unique_subjects, unique_ages], names=['src_subject_id', 'Age'])
    full_data = pd.DataFrame(index=all_combinations).reset_index()

    # Step 3: Extract baseline data for filling
    baseline_data = df[df['eventname'] == 'baseline_year_1_arm_1'].set_index('src_subject_id')[effects]

    # Step 4: Merge the full data table with baseline data to fill in information for each src_subject_id
    df_prediction = full_data.merge(baseline_data, left_on='src_subject_id', right_index=True, how='left')

    # Step 5: Ensure columns are in the same order as df
    df_prediction = df_prediction[df.columns.intersection(df_prediction.columns)]

    # Generate Time interaction effects
    effects_of_interest = [f'Age:{effect}' for effect in effects]
    
    # Initialize dictionary to collect p-values for FDR correction
    p_values_dict = {effect: [] for effect in effects_of_interest}
    
    # Build the model formula dynamically
    fixed_effects = ' + '.join([f'Age * {effect}' for effect in effects])
    random_effects = '(1 | src_subject_id)'
    anova_results = {}
    
    for key in df.columns:
        if not pd.api.types.is_numeric_dtype(df[key]) or key in ['Subtype', 
                    'Age', 'Time','Sex','Parental_Education', 'Income', 'Marital', 'ICV', 'Treatment']:
            continue
        print(f'Analyzing {key}...')
        
        # Construct the complete model formula
        model_formula = f"{key} ~ {fixed_effects} + {random_effects}"
        
        # Fit the model
        model = Lmer(model_formula, data=df)
        model.fit(REML=True, summary=False)
        
        # Get ANOVA results
        anova_df = model.anova()

        # Generate predictions
        df_prediction[f'{key}'] = model.predict(df_prediction, skip_data_checks=True, verify_predictions=False)

        # Calculate partial eta squared
        partial_eta_sq = calculate_partial_eta_squared(anova_df, effects_of_interest)
        
        # Get p-values for effects of interest
        p_values = anova_df.loc[effects_of_interest, 'P-val']
        
        # Combine p-values and partial eta squared
        results_dict = {effect: {"p_value": p_values[effect],
                                 "eta_sq": partial_eta_sq[effect]} 
                       for effect in effects_of_interest}
        
        anova_results[key] = results_dict
        
        # Collect p-values for each effect
        for effect in effects_of_interest:
            p_values_dict[effect].append(p_values[effect])

    # Apply FDR correction for each effect separately
    for effect, p_values in p_values_dict.items():
        corrected_p_values = multipletests(p_values, method='fdr_bh')[1]
        # Update anova_results with corrected p-values
        for (key, results), corrected_p in zip(anova_results.items(), corrected_p_values):
            results[effect]["p_value"] = corrected_p

    # Create row indices for the final DataFrame
    row_indices = []
    for effect in effects_of_interest:
        row_indices.extend([f'{effect}_p_fdr', f'{effect}_eta_sq'])

    # Convert anova_results to a DataFrame
    data = {}
    for key in anova_results:
        values = []
        for effect in effects_of_interest:
            values.extend([
                anova_results[key][effect]["p_value"],
                anova_results[key][effect]["eta_sq"]
            ])
        data[key] = values

    # Create the final DataFrame with dynamic effects as rows
    df_anova = pd.DataFrame(data, index=row_indices)
    
    # Adjust 'Age' in df_prediction
    df_prediction['Age'] = df_prediction['Age'] // 12
    
    return df_prediction, df_anova


def calculate_partial_eta_squared(anova_df, effects):
    """
    Calculate partial eta squared for specified effects in ANOVA results
    
    Parameters:
    -----------
    anova_df : pandas.DataFrame
        ANOVA results dataframe with following columns:
        - SS (Sum of Squares)
        - NumDF (Numerator Degrees of Freedom) 
        - DenomDF (Denominator Degrees of Freedom)
        - F-stat (F statistics)
    effects : str or list
        Name(s) of the effect(s) to calculate, must be in the indices of anova_df
        Can be either a single string or a list of strings
        
    Returns:
    --------
    dict if multiple effects, float if single effect
        Partial eta squared values for each effect
        If single effect provided as string, returns float
        If list of effects provided, returns dict with effects as keys
        
    Raises:
    -------
    ValueError
        If any of the specified effects are not found in the ANOVA results
        If required columns are missing from the ANOVA results dataframe
    """
    # Verify required columns exist
    required_cols = ['SS', 'NumDF', 'DenomDF', 'F-stat']
    missing_cols = [col for col in required_cols if col not in anova_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in ANOVA results: {missing_cols}")
    
    # Convert single effect to list for consistent processing
    if isinstance(effects, str):
        effects = [effects]
        return_single = True
    else:
        return_single = False
        
    # Check if all effects exist in the dataframe
    missing_effects = [effect for effect in effects if effect not in anova_df.index]
    if missing_effects:
        raise ValueError(f"Effects {missing_effects} not found in ANOVA results")
    
    results = {}
    
    for effect in effects:
        try:
            # Extract required values
            SS_effect = anova_df.loc[effect, 'SS']
            df_effect = anova_df.loc[effect, 'NumDF']
            df_error = anova_df.loc[effect, 'DenomDF']
            F_stat = anova_df.loc[effect, 'F-stat']
            
            # Validate numeric values
            if any(pd.isna([SS_effect, df_effect, df_error, F_stat])):
                raise ValueError(f"Missing or NaN values found for effect {effect}")
            
            # Calculate Mean Square for effect
            MS_effect = SS_effect / df_effect
            
            # Derive Mean Square error from F-statistic
            MS_error = MS_effect / F_stat
            
            # Calculate Sum of Squares error
            SS_error = MS_error * df_error
            
            # Calculate partial eta squared
            partial_eta_sq = SS_effect / (SS_effect + SS_error)
            
            results[effect] = partial_eta_sq
            
        except Exception as e:
            raise ValueError(f"Error calculating partial eta squared for effect {effect}: {str(e)}")
    
    # Return single value if input was single string
    return results[effects[0]] if return_single else results


def calculate_correlations(ids, df_value, df_demo, demo_column):
    """
    Calculate Pearson correlations between a demographic column and brain imaging features.
    
    Parameters:
    -----------
    df_value : pandas.DataFrame
        DataFrame containing brain imaging features with subject IDs as index
    df_demo : pandas.DataFrame
        DataFrame containing demographic information with subject IDs as index
    demo_column : str
        Name of the demographic column to use for correlation analysis
    
    Returns:
    --------
    pandas.DataFrame
        Correlation results with columns: 
        - correlation: Pearson correlation coefficient
        - p_value: p-value of the correlation
    """
    ids = pd.Index(ids).intersection(df_value.index).intersection(df_demo.index)

    # Find indices with non-empty values in the specified demographic column
    valid_indices = df_demo.loc[ids][demo_column].dropna().index
    
    # Subset both dataframes to common indices
    brain_subset = df_value.loc[valid_indices]
    demo_subset = df_demo.loc[valid_indices]
    
    # Prepare results storage
    correlations = {}
    
    # Perform correlation for each brain feature
    for var in df_value.columns:
        # Extract the column for correlation
        demo_values = demo_subset[demo_column]
        values = brain_subset[var]
        
        # Calculate Pearson correlation
        correlation, p_value = pearsonr(demo_values, values)
        
        correlations[var] = {
            'correlation': correlation,
            'p_value': p_value
        }
    
    # Convert to DataFrame for easy viewing
    corr_df = pd.DataFrame.from_dict(correlations, orient='index')
    
    # Sort by absolute correlation value
    corr_df = corr_df.reindex(corr_df['correlation'].abs().sort_values(ascending=False).index)
    
    return corr_df

def analyze_correlations(
    ids: np.ndarray,
    ids_tc: np.ndarray, 
    labels: np.ndarray,
    demo: pd.DataFrame,
    df_y: pd.DataFrame,
    df_x: pd.DataFrame,
    covariants: List[str] = ['sex', 'age', 'parental_education', 'income', 'marital'],
    factor: List[str] = None,
    fdr: Literal['all', 'factor'] = 'all',
) -> Dict[str, pd.DataFrame]:
    
    ids_ = pd.Index(ids).intersection(df_x.index).intersection(df_y.index).intersection(demo.index)
    labels = labels[np.isin(ids, ids_)]
    ids_tc = pd.Index(ids_tc).intersection(df_x.index).intersection(df_y.index).intersection(demo.index)

    if factor is None:
        factor = df_x.columns.tolist()
    
    all_ids = {
        'TC': ids_tc,
        'IA': ids_[labels == 0],
        'HF': ids_[labels == 1], 
        'ME': ids_[labels == 2]
    }
    
    def partial_correlation(x: pd.Series, y: pd.Series, covs: pd.DataFrame = None) -> tuple:
        if covs is None or covs.empty:
            valid_idx = pd.concat([x, y], axis=1).dropna().index
            if len(valid_idx) == 0:
                return np.nan, np.nan, 0
            
            x_vals = np.array(x[valid_idx]).flatten()
            y_vals = np.array(y[valid_idx]).flatten()
            
            unique_vals = np.unique(x_vals)
            if len(unique_vals) == 1:
                return np.nan, np.nan, len(valid_idx)
            r, p = pearsonr(x_vals, y_vals)
            return r, p, len(valid_idx)
            
        valid_idx = pd.concat([x, y, covs], axis=1).dropna().index
        if len(valid_idx) == 0:
            return np.nan, np.nan, 0
            
        x = np.array(x[valid_idx]).reshape(-1, 1)
        y = np.array(y[valid_idx]).reshape(-1, 1)
        covs = np.array(covs.loc[valid_idx])
        
        reg_x = LinearRegression().fit(covs, x)
        x_resid = x - reg_x.predict(covs)
        
        reg_y = LinearRegression().fit(covs, y)
        y_resid = y - reg_y.predict(covs)
        
        unique_vals = np.unique(x_resid)
        if len(unique_vals) == 1:
            return np.nan, np.nan, len(valid_idx)
        r, p = pearsonr(x_resid.flatten(), y_resid.flatten())
            
        return r, p, len(valid_idx)

    def compare_r_values(r1: float, r2: float, n1: int, n2: int) -> float:
        z1 = np.arctanh(r1)
        z2 = np.arctanh(r2)
        se = np.sqrt(1/(n1-3) + 1/(n2-3))
        z = (z1 - z2) / se
        p = 2 * (1 - norm.cdf(abs(z)))
        return p
    
    results = {}
    
    for f in factor:
        result_df = pd.DataFrame(
            index=['TC_r', 'TC_p', 'TC_d',
                  'IA_r', 'IA_p', 'IA_d', 'IA_sig',
                  'HF_r', 'HF_p', 'HF_d', 'HF_sig',
                  'ME_r', 'ME_p', 'ME_d', 'ME_sig'],
            columns=df_y.columns
        )
        
        r_diff_ps = {
            'IA': [],
            'HF': [],
            'ME': []
        }
        
        corr_ps = []
        corr_ps_indices = []
        factor_corr_ps = {group: [] for group in ['TC', 'IA', 'HF', 'ME']}
        factor_corr_indices = {group: [] for group in ['TC', 'IA', 'HF', 'ME']}
        
        for y_col in df_y.columns:
            tc_data = {}
            for group, group_ids in all_ids.items():
                common_idx = pd.Index(group_ids).intersection(df_x.index).intersection(df_y.index)
                if covariants:
                    common_idx = common_idx.intersection(demo.index)
                    cov_data = demo.loc[common_idx, covariants]
                else:
                    cov_data = None
                    
                x_data = df_x.loc[common_idx, f]
                y_data = df_y.loc[common_idx, y_col]
                
                r, p, n = partial_correlation(x_data, y_data, cov_data)
                
                result_df.loc[f'{group}_r', y_col] = r
                if not np.isnan(p):
                    corr_ps.append(p)
                    corr_ps_indices.append((group, y_col))
                    factor_corr_ps[group].append(p)
                    factor_corr_indices[group].append(y_col)
                # cohen's d
                d = 2 * r / np.sqrt(1 - r**2) if abs(r) < 1 else np.nan
                result_df.loc[f'{group}_d', y_col] = d
                
                if group == 'TC':
                    tc_data = {'r': r, 'n': n}
                else:
                    if not np.isnan(r) and not np.isnan(tc_data['r']):
                        p_diff = compare_r_values(r, tc_data['r'], n, tc_data['n'])
                        r_diff_ps[group].append(p_diff)
                    else:
                        r_diff_ps[group].append(np.nan)
        
        # FDR correction
        if fdr == 'all' and corr_ps:
            # correct all p values
            _, fdr_ps, _, _ = multipletests(corr_ps, method='fdr_bh')
            for (group, y_col), p_corrected in zip(corr_ps_indices, fdr_ps):
                result_df.loc[f'{group}_p', y_col] = p_corrected
        elif fdr == 'factor' and corr_ps:
            # correct p values for each factor
            _, fdr_ps, _, _ = multipletests(corr_ps, method='fdr_bh')
            for (group, y_col), p_corrected in zip(corr_ps_indices, fdr_ps):
                result_df.loc[f'{group}_p', y_col] = p_corrected
        
        # compare IA, HF, ME with TC 
        for group in ['IA', 'HF', 'ME']:
            valid_ps = [p for p in r_diff_ps[group] if not np.isnan(p)]
            if valid_ps:
                _, fdr_ps, _, _ = multipletests(valid_ps, method='fdr_bh')
                curr_idx = 0
                for i, p in enumerate(r_diff_ps[group]):
                    if not np.isnan(p):
                        p_corrected = fdr_ps[curr_idx]
                        curr_idx += 1
                        
                        tc_r = result_df.loc['TC_r', df_y.columns[i]]
                        group_r = result_df.loc[f'{group}_r', df_y.columns[i]]
                        
                        if abs(group_r) > abs(tc_r):
                            if p_corrected < 0.001:
                                result_df.loc[f'{group}_sig', df_y.columns[i]] = '***'
                            elif p_corrected < 0.01:
                                result_df.loc[f'{group}_sig', df_y.columns[i]] = '**'
                            elif p_corrected < 0.05:
                                result_df.loc[f'{group}_sig', df_y.columns[i]] = '*'
                            else:
                                result_df.loc[f'{group}_sig', df_y.columns[i]] = ''
                        else:
                            result_df.loc[f'{group}_sig', df_y.columns[i]] = ''
        
        results[f] = result_df
    
    return results



def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    r, p = pearsonr(y_true, y_pred)
    return {
        'r2': r2,
        'r': r,
        'p': p
    }

def aggregate_predictions_by_id(ids, y_true, y_pred):
    df = pd.DataFrame({
        'id': ids,
        'y_true': y_true,
        'y_pred': y_pred
    })
    agg_df = df.groupby('id').mean()
    return agg_df['y_true'].values, agg_df['y_pred'].values

def process_cv_results(all_fold_predictions, args):
    n_targets = args.Y.shape[1]
    results = {}
    
    all_ids = []
    all_true = []
    all_pred = []
    
    # Collect all predictions
    for fold_pred in all_fold_predictions:
        all_ids.extend(fold_pred['ids'])
        all_true.append(fold_pred['true_values'])
        all_pred.append(fold_pred['predictions'])
    
    all_true = np.vstack(all_true)
    all_pred = np.vstack(all_pred)
    
    # Initialize aggregated arrays for all targets
    agg_true_all = []
    agg_pred_all = []
    mean_r2 = 0
    mean_r = 0

    for target_idx in range(n_targets):
        agg_true, agg_pred = aggregate_predictions_by_id(
            all_ids,
            all_true[:, target_idx],
            all_pred[:, target_idx]
        )
        
        # Store aggregated values for this target
        agg_true_all.append(agg_true)
        agg_pred_all.append(agg_pred)
        
        metrics = calculate_metrics(agg_true, agg_pred)
        results[SCALES[target_idx]] = metrics
        
        mean_r2 += metrics['r2']
        mean_r += metrics['r']
    
    results['mean'] = {
        'r2': mean_r2 / n_targets,
        'r': mean_r / n_targets,
        'p': None
    }
    
    # Convert lists to numpy arrays and transpose to match original shape
    agg_true_all = np.array(agg_true_all).T
    agg_pred_all = np.array(agg_pred_all).T
    
    return results, agg_true_all, agg_pred_all

def process_test_results(test_predictions, args):
    n_targets = args.Y.shape[1]
    results = {}
    
    # Initialize aggregated arrays for all targets
    agg_true_all = []
    agg_pred_all = []
    mean_r2 = 0
    mean_r = 0
    
    for target_idx in range(n_targets):
        agg_true, agg_pred = aggregate_predictions_by_id(
            test_predictions['ids'],
            test_predictions['true_values'][:, target_idx],
            test_predictions['predictions'][:, target_idx]
        )
        
        # Store aggregated values for this target
        agg_true_all.append(agg_true)
        agg_pred_all.append(agg_pred)
        
        metrics = calculate_metrics(agg_true, agg_pred)
        results[SCALES[target_idx]] = metrics
        
        mean_r2 += metrics['r2']
        mean_r += metrics['r']
    
    results['mean'] = {
        'r2': mean_r2 / n_targets,
        'r': mean_r / n_targets,
        'p': None
    }
    
    # Convert lists to numpy arrays and transpose to match original shape
    agg_true_all = np.array(agg_true_all).T
    agg_pred_all = np.array(agg_pred_all).T
    
    return results, agg_true_all, agg_pred_all

def results_to_df(results):
    rows = []
    for scale in results.keys():
        metrics = results[scale]
        rows.append({
            'Scale': scale,
            'R2': metrics['r2'],
            'Pearson_r': metrics['r'],
            'P_value': metrics['p'] if scale != 'mean' else None
        })
    return pd.DataFrame(rows)


def deconfound(X, confounders, demean_confounders=False):
    """Deconfound a data matrix.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        data matrix to be confounded
    confounders : np.ndarray (n_samples, n_confounds)
        confound matrix
    demean_confounders : bool
        if ``True`` demeaned confounders are used

    Returns
    -------
    deconfounded : np.ndarray (n_samples, n_features)
        deconfounded data matrix
    """
    if confounders.shape[1] == 0:  # nothing to do
        return X
    if demean_confounders:
        confounders = confounders - confounders.mean(0, keepdims=True)
    deconfound_beta = np.linalg.pinv(confounders) @ X
    return X - confounders @ deconfound_beta


def rank_based_inverse_normal_transform(x, c='blom'):
    """Rank-based inverse normal transformation.

    References
    ----------
    Beasley TM, Erickson S, Allison DB. Rank-Based Inverse Normal
        Transformations are Increasingly Used, But are They Merited?
        Behav Genet. 2009 Jun 14;39(5):580.


    Parameters
    ----------
    x : np.ndarray-like (n_samples,) or (n_samples x n_features)
        data to be transformed
    c : float or str
        constant appearing in transformation, named values (cf. Beasley et al.)
        are
        - 'blom': c = 3/8
        - 'tukey': c = 1/3
        - 'rankit' c = 1/2
        - 'waerden' c = 0

    Returns
    -------
    x_INT : (n_samples,) or (n_samples x n_features)
        transformed data
    """
    
    c = dict(
        blom=3./8,
        tukey=1./3,
        rankit=1./2,
        waerden=0
    ).get(c, c)

    x = np.asarray(x)

    is_finite = np.isfinite(x)
    ranks = np.where(
        is_finite,
        np.argsort(np.argsort(x, axis=0), axis=0) + 1,
        np.nan
    )

    z = (ranks - c) / (is_finite.sum(0) - 2*c + 1)
    if is_finite.all():
        x_INT = norm.ppf(z)
    else:  # loop as workaround to suppress runtime warnings
        x_INT = np.nan * np.empty_like(z)
        for i in range(z.size):
            if np.isfinite(z.flat[i]):
                x_INT.flat[i] = norm.ppf(z.flat[i])

    assert np.allclose(np.mean(np.ma.masked_invalid(x_INT), axis=0), 0)

    return x_INT




# ---------- Formatting helpers ----------
def fmt3(x):
    """Format a number with 3 significant digits."""
    try:
        return f"{float(x):.3g}"
    except Exception:
        return ""

def fmt_vec(arr, k=None):
    """Format a 1D vector; if k is given, show first k with an ellipsis."""
    a = np.asarray(arr).reshape(-1)
    if k is None or len(a) <= k:
        return "[" + ", ".join(fmt3(v) for v in a) + "]"
    return "[" + ", ".join(fmt3(v) for v in a[:k]) + ", ...]"


# # ---------- splits folds to even Y  ----------
# def _zscore_cols(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
#     """Column-wise z-score with small ridge to avoid zero std."""
#     A = np.asarray(A, float)
#     mu = A.mean(axis=0, keepdims=True)
#     sd = A.std(axis=0, ddof=1, keepdims=True)
#     sd[sd < eps] = eps
#     return (A - mu) / sd

# def _quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
#     """Quantile binning to integer labels in [0, n_bins-1]."""
#     # Use unique edges to avoid duplicate cut points in small samples
#     qs = np.linspace(0, 1, n_bins + 1)
#     edges = np.unique(np.quantile(x, qs))
#     # Ensure at least 2 edges
#     if edges.size < 2:
#         return np.zeros_like(x, dtype=int)
#     # Map to bins; rightmost inclusive
#     bins = np.digitize(x, edges[1:-1], right=False)
#     # Clip to [0, n_bins-1] in case of numerical issues
#     bins = np.clip(bins, 0, len(edges) - 2)
#     return bins.astype(int)

# Deterministic stratified splits for continuous (or multivariate) Y.
# No implicit fallback, no hidden randomness except seed.
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import PCA

def _make_bins_from_Y(Y, y_bins=5, y_pc=2, *, cv_seed):
    """
    Deterministically create integer labels from continuous / multivariate Y.
    Strategy:
      - If Y is multivariate, reduce to `y_pc` principal components (deterministic PCA with random_state=cv_seed).
      - For the first (primary) PC, use quantile binning into y_bins equal-frequency bins via np.percentile.
    Returns 1D integer labels in [0, y_bins-1].
    Raises ValueError if y_bins < 2 or if Y has too few unique values to form bins.
    """
    Y = np.asarray(Y)
    if Y.ndim == 1:
        vec = Y.astype(float).ravel()
    else:
        # deterministic PCA (fix random_state)
        pca = PCA(n_components=min(y_pc, Y.shape[1]), random_state=int(cv_seed))
        pcs = pca.fit_transform(Y)
        vec = pcs[:, 0].ravel()

    if y_bins < 2:
        raise ValueError("y_bins must be >= 2")
    if np.allclose(vec, vec[0]):
        # all identical -> single label
        return np.zeros(vec.shape[0], dtype=int)

    # quantile bin edges (includes 0 and 100)
    perc = np.linspace(0, 100, num=y_bins + 1)
    edges = np.percentile(vec, perc)
    # ensure strictly increasing edges by tiny eps where needed
    for i in range(1, edges.size):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-8
    # digitize -> returns 1..y_bins, map to 0..y_bins-1
    labels = np.digitize(vec, edges[1:-1], right=True).astype(int)
    return labels

def make_splits_stratified_by_Y(ids: np.ndarray,
                                Y: np.ndarray,
                                nfold: int,
                                cv_seed: int,
                                y_bins: int = 5,
                                y_pc: int = 2) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Deterministic stratified splits.
    - Builds labels via _make_bins_from_Y (quantile binning on first PC if multivariate).
    - Uses StratifiedKFold(shuffle=True, random_state=cv_seed).
    - If any label has count < nfold, raises ValueError (no silent fallback).
    Returns list of (train_idx, valid_idx) where indices are positions 0..N-1.
    """
    ids = np.asarray(ids)
    N = ids.shape[0]
    if N < nfold:
        raise ValueError(f"N={N} < nfold={nfold}")

    labels = _make_bins_from_Y(Y, y_bins=y_bins, y_pc=y_pc, cv_seed=cv_seed)
    uniq, cnt = np.unique(labels, return_counts=True)
    if (cnt < nfold).any():
        # fail loudly so user can decide (you said你不怕报错)
        bad = {int(u): int(c) for u, c in zip(uniq, cnt) if c < nfold}
        raise ValueError(f"Some label bins have fewer than nfold members: {bad}. "
                         "Reduce nfold or reduce y_bins / adjust y_pc to ensure >= nfold per bin.")

    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=int(cv_seed))
    splits = []
    for tr_idx, va_idx in skf.split(np.zeros(N), labels):
        splits.append((tr_idx, va_idx))
    return splits


# ---------- Feature preprocessing ----------
def preprocess_feat(A, mode="norm"):
    """
    Returns (A_processed, mu, sd)
    'none'   : no-op
    'demean' : subtract mean
    'norm'   : z-score
    """
    if mode is None or str(mode).lower() == "none":
        mu = np.zeros((1, A.shape[1]))
        sd = np.ones((1, A.shape[1]))
        return A, mu, sd

    m = str(mode).lower()
    mu = A.mean(0, keepdims=True)
    A2 = A - mu

    if m == "demean":
        sd = np.ones((1, A.shape[1]))
        return A2, mu, sd

    if m == "norm":
        sd = A2.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        return A2 / sd, mu, sd

    raise ValueError(f"Unknown preprocessing mode: {mode}")

# ---------- PCA reference ----------
def pca_reference(X, Y, nPC, xnorm="norm", ynorm="norm", whiten=False):
    """PCA on full X; Y processed separately."""
    Y_proc, _, _ = preprocess_feat(Y, ynorm)
    X_proc, _, _ = preprocess_feat(X, xnorm)

    pca = PCA(n_components=nPC, svd_solver="full")
    Fe = pca.fit_transform(X_proc)
    evr_sum = float(pca.explained_variance_ratio_.sum())

    if whiten:
        pc_sd = Fe.std(0); pc_sd[pc_sd == 0] = 1.0
        Fe_out = Fe / pc_sd
    else:
        pc_sd = np.ones(Fe.shape[1])
        Fe_out = Fe.copy()

    return dict(
        Fe_s=Fe_out,
        Y_z=Y_proc,
        V=pca.components_,
        pc_sd=pc_sd,
        explained_var_sum=evr_sum
    )


# ---------- PCA CV cache ----------
def pca_cv_cache(
    X, Y, ids, nfold, cv_seed, nPC,
    xnorm="norm", ynorm="norm", whiten=False,
    y_bins=5, y_pc=2
):
    """CV PCA: train-fit PCA; val uses train's mu/sd."""
    splits = make_splits_stratified_by_Y(
        np.asarray(ids), Y,
        nfold=nfold, cv_seed=cv_seed,
        y_bins=y_bins, y_pc=y_pc
    )

    cache = []
    for tr_idx, va_idx in splits:
        tr_idx = np.asarray(tr_idx, int)
        va_idx = np.asarray(va_idx, int)

        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]

        # ----- Y preprocess (TRAIN stats) -----
        Yz_tr, mu_y, sd_y = preprocess_feat(Y_tr, ynorm)
        Yz_va = (Y_va - mu_y) / sd_y

        # ----- X preprocess (TRAIN stats) -----
        X_tr_proc, mu_x, sd_x = preprocess_feat(X_tr, xnorm)
        X_va_proc = (X_va - mu_x) / sd_x

        # ----- PCA on TRAIN -----
        pca = PCA(n_components=nPC, svd_solver="full")
        Fe_tr = pca.fit_transform(X_tr_proc)

        if whiten:
            pc_sd = Fe_tr.std(0); pc_sd[pc_sd == 0] = 1.0
            Fe_tr_out = Fe_tr / pc_sd
        else:
            pc_sd = np.ones(Fe_tr.shape[1])
            Fe_tr_out = Fe_tr.copy()

        cache.append(dict(
            tr_idx=tr_idx, va_idx=va_idx, ids_va=np.asarray(ids[va_idx], int),
            X_va=X_va_proc, Y_va=Yz_va,
            Fe_tr_s=Fe_tr_out, Yz_tr=Yz_tr,
            V_tr=pca.components_, pc_sd=pc_sd,
            mu_y=mu_y, sd_y=sd_y, mu_x=mu_x, sd_x=sd_x
        ))
    return cache



def CCA_align(U_ref, U_cv, U_fc_cv, fc_score_list, behav_score_list):
    """
    Align CV fold loadings/scores to reference model.
    Performs:
        1) Hungarian dimension matching
        2) loading-level sign alignment (to ref)
        3) score-level sign alignment (to ref)
        4) concat all folds
    """

    n_folds, _, n_dims = U_cv.shape

    # aligned versions
    U_cv_al      = U_cv.copy()
    U_fc_cv_al   = U_fc_cv.copy()
    fc_score_al  = [s.copy() for s in fc_score_list]
    behav_score_al = [s.copy() for s in behav_score_list]

    for f in range(n_folds):

        # correlation matrix ref ↔ cv
        corr_mat = np.corrcoef(U_ref.T, U_cv[f].T)
        corr_mat = corr_mat[:n_dims, n_dims:]     # shape (D,D)

        # Hungarian
        _, match = linear_sum_assignment(-np.abs(corr_mat))

        # loading sign from REF
        signs = np.sign(corr_mat[np.arange(n_dims), match])
        signs[signs == 0] = 1

        # apply loading flip
        U_cv_al[f]      = U_cv[f][:, match]      * signs
        U_fc_cv_al[f]   = U_fc_cv[f][:, match]   * signs

        # apply score flip
        fc_score_al[f]    = fc_score_list[f][:, match]    * signs
        behav_score_al[f] = behav_score_list[f][:, match] * signs

    # ---------- concat folds ----------
    fc_score_all    = np.concatenate(fc_score_al, axis=0)
    behav_score_all = np.concatenate(behav_score_al, axis=0)

    return U_cv_al, U_fc_cv_al, fc_score_all, behav_score_all


# ---------- Sign fixing ----------
def per_dim_sign_fix_noW(U_ref, U_fc_ref, U_cv, U_fc_cv,
                         fc_score, behav_score, D_eval=2):
    """
    Only enforce a consistent sign convention for reference model.
    CV outputs are only flipped when ref is flipped.
    Does NOT use r_cv or validation data in any way.
    """

    U_ref       = U_ref.copy()
    U_fc_ref    = U_fc_ref.copy()
    U_cv        = U_cv.copy()
    U_fc_cv     = U_fc_cv.copy()
    fc_score    = fc_score.copy()
    behav_score = behav_score.copy()

    D = U_ref.shape[1]
    D_use = min(D_eval, D)

    for d in range(D_use):
        col = U_ref[:, d]

        # if main direction is negative, flip all consistently
        if -col.min() > col.max():
            U_ref[:, d]       *= -1
            U_fc_ref[:, d]    *= -1
            U_cv[:, :, d]     *= -1
            U_fc_cv[:, :, d]  *= -1
            fc_score[:, d]    *= -1
            behav_score[:, d] *= -1

    return U_ref, U_fc_ref, U_cv, U_fc_cv, fc_score, behav_score


# def per_dim_sign_fix_noW(Ub_ref, Uc_ref, U_behav_cv, U_fc_cv, fc_cv, behav_cv, D_eval=2):
#     """
#     Flip sign per dimension if needed based on ref behavioral loadings.
#     Mutates copies of arrays and returns fixed versions.
#     """
#     D = Ub_ref.shape[1]
#     for d in range(min(D_eval, D)):
#         if -Ub_ref[:, d].min() > Ub_ref[:, d].max():
#             Ub_ref[:, d]        *= -1.0
#             Uc_ref[:, d]        *= -1.0
#             U_behav_cv[:, :, d] *= -1.0
#             U_fc_cv[:, :, d]    *= -1.0
#             fc_cv[:, d]         *= -1.0
#             behav_cv[:, d]      *= -1.0
#     return Ub_ref, Uc_ref, U_behav_cv, U_fc_cv, fc_cv, behav_cv

def get_data(scale_collection,
             npz_path: str = './data/Iowa_Lesion_MMPI.npz',
             csv: str = './mmpi/Iowa_MMPI2RF_byNandi.csv'):

    # --- load cognition CSV ---
    cog_raw = pd.read_csv(csv)

    # detect ID column
    if 'study_id' in cog_raw.columns:
        id_col = 'study_id'
    elif 'ID' in cog_raw.columns:
        id_col = 'ID'
    else:
        raise ValueError

    # --- set index and convert to int (no pad, no strip) ---
    cog = cog_raw.set_index(id_col)
    cog.index = cog.index.astype(int)

    # 1) KEEP == 1
    cog = cog[cog["KEEP"] == 1].copy()

    # 2) keep only scales (NO LONGER require all keys to exist)
    keep_cols = [c for c in scale_collection if c in cog.columns]   # ← 修改点 1
    cog_tmp = cog[keep_cols].copy()                                # ← 修改点 2

    # 3) remove rows containing 999
    cog_tmp = cog_tmp[~(cog_tmp == 999).any(axis=1)].copy()

    # final behavior table before matching
    cog_sel = cog_tmp

    # -----------------------------
    # ID matching (pure int version)
    # -----------------------------
    behav_ids_int = cog_sel.index.astype(int)

    npz = np.load(npz_path, allow_pickle=True)
    file_ids_int = np.asarray(npz["ids"]).astype(int)

    # intersection on pure integers
    common_ids_int = sorted(set(behav_ids_int) & set(file_ids_int))

    matched_ids = np.array(common_ids_int, dtype=int)

    # map npz row indices
    id_to_idx = {fid: i for i, fid in enumerate(file_ids_int)}
    idx = [id_to_idx[mid] for mid in matched_ids]

    # --- subset brain features (allow missing keys) ---
    possible_keys = [
        'sfc_ml_pr',
        'sfc_ml_sp',
        'sfc_nc_pr',
        'sfc_nc_sp',
        'sc_ml',
        'sc_nc',
        'fc',
    ]

    data = {}
    for k in possible_keys:
        if k in npz:
            arr = npz[k][idx].astype(np.float32)
            data[k] = arr
        else:
            # skip missing keys silently
            pass

    # --- behavior ---
    behav = cog_sel.loc[matched_ids].values.astype(np.float32)

    return matched_ids, data, behav

# def get_data(scale_collection,
#              npz_path: str = './data/Iowa_Lesion.npz',
#              csv: str = './cognition/final_train_les_demog25.csv'):
#     """
#     Loads cognition + lesion features, applies:
#       1) KEEP == 1
#       2) Select only scale_collection columns
#       3) Remove rows containing 999
#       4) Drop NaNs
#       5) Align with NPZ ids
#       6) Return float32 brain + behavior
#     """

#     # --- load cognition CSV ---
#     cog_raw = pd.read_csv(csv)

#     # detect ID column
#     if 'study_id' in cog_raw.columns:
#         id_col = 'study_id'
#     elif 'ID' in cog_raw.columns:
#         id_col = 'ID'
#     else:
#         raise ValueError(f"CSV {csv} must contain either 'study_id' or 'ID' column.")

#     cog = cog_raw.set_index(id_col)

#     # --------------------------------------------------------
#     # 1) KEEP == 1   （你要求的第一步）
#     # --------------------------------------------------------
#     if 'KEEP' in cog.columns:
#         cog = cog[cog['KEEP'] == 1].copy()
#     else:
#         raise ValueError("CSV must contain column 'KEEP' for step 1 filtering.")

#     # --------------------------------------------------------
#     # 2) 只保留 scale_collection   （你当作 scale_name_map.keys()）
#     # --------------------------------------------------------
#     # 注意：如果 scale_collection 是 dict.keys，需要转换为 list
#     cols = list(scale_collection)
#     cog_tmp = cog[cols].copy()
#     # --------------------------------------------------------
#     # 3) 去除含 999 的行（任意列含 999 均去除）
#     # --------------------------------------------------------
#     cog_tmp = cog_tmp[~(cog_tmp == 999).any(axis=1)].copy()
#     # --------------------------------------------------------
#     # 4) 去除 NaN
#     # --------------------------------------------------------
#     cog_sel = cog_tmp.dropna()

#     table_ids = cog_sel.index.astype(int).values

#     # --- load NPZ ---
#     npz = np.load(npz_path, allow_pickle=True)
#     file_ids = np.asarray(npz['ids'])

#     if file_ids.dtype.type in (np.bytes_, np.str_):
#         file_ids = np.array([int(x.decode() if isinstance(x, bytes) else x)
#                              for x in file_ids], dtype=int)
#     else:
#         file_ids = file_ids.astype(int)

#     id_to_idx = {int(fid): i for i, fid in enumerate(file_ids)}

#     idx = [id_to_idx[sid] for sid in table_ids if int(sid) in id_to_idx]
#     matched_ids = np.array([sid for sid in table_ids if int(sid) in id_to_idx], dtype=int)

#     # --- subset brain features ---
#     data = {
#         'sfc_ml_pr': npz['sfc_ml_pr'][idx],
#         'sfc_ml_sp': npz['sfc_ml_sp'][idx],
#         'sfc_nc_pr': npz['sfc_nc_pr'][idx],
#         'sfc_nc_sp': npz['sfc_nc_sp'][idx],
#         'sc_ml':     npz['sc_ml'][idx],
#         'sc_nc':     npz['sc_nc'][idx],
#         'fc':        npz['fc'][idx],
#     }

#     # --- behavior ---
#     behav = cog_sel.loc[matched_ids].values

#     # --- ensure float32 ---
#     behav = behav.astype(np.float32)
#     for k in data:
#         data[k] = data[k].astype(np.float32)

#     return matched_ids, data, behav



def balance_masks(train_mask, test_mask, cog, cog_test, alpha=0.05, min_train=10, min_test=10, top_k=50, max_iter=2000):
    tm = train_mask.copy()
    sm = test_mask.copy()

    def pvals(tm, sm):
        a1, a2 = cog.loc[tm,'age'], cog_test.loc[sm,'age']
        e1, e2 = cog.loc[tm,'educ'], cog_test.loc[sm,'educ']
        if len(a1)<2 or len(a2)<2 or len(e1)<2 or len(e2)<2:
            return 0,0
        pa = ttest_ind(a1,a2,equal_var=False).pvalue
        pe = ttest_ind(e1,e2,equal_var=False).pvalue
        return pa if not np.isnan(pa) else 0, pe if not np.isnan(pe) else 0

    pa, pe = pvals(tm, sm)
    it = 0
    while (pa < alpha or pe < alpha) and tm.sum()>min_train and sm.sum()>min_test and it<max_iter:
        it += 1
        tr, te = cog.loc[tm,['age','educ']], cog_test.loc[sm,['age','educ']]
        tr_ma, tr_sa = tr['age'].mean(), tr['age'].std(ddof=1) or 1
        tr_me, tr_se = tr['educ'].mean(), tr['educ'].std(ddof=1) or 1
        te_ma, te_sa = te['age'].mean(), te['age'].std(ddof=1) or 1
        te_me, te_se = te['educ'].mean(), te['educ'].std(ddof=1) or 1

        tr['score'] = np.maximum(abs((tr['age']-te_ma)/te_sa), abs((tr['educ']-te_me)/te_se))
        te['score'] = np.maximum(abs((te['age']-tr_ma)/tr_sa), abs((te['educ']-tr_me)/tr_se))

        tr_idx = tr.sort_values('score', ascending=False).head(top_k).index
        te_idx = te.sort_values('score', ascending=False).head(top_k).index

        best, best_metric = None, -1
        cur_metric = min(pa, pe)
        for idx in tr_idx:
            tmp = tm.copy(); tmp.loc[idx] = False
            p1,p2 = pvals(tmp, sm); m = min(p1,p2)
            if m>best_metric: best_metric, best = m, ('train', idx, p1, p2)
        for idx in te_idx:
            tmp = sm.copy(); tmp.loc[idx] = False
            p1,p2 = pvals(tm, tmp); m = min(p1,p2)
            if m>best_metric: best_metric, best = m, ('test', idx, p1, p2)
        if best is None or best_metric <= cur_metric + 1e-12: break
        side, idx, pa, pe = best
        if side == 'train': tm.loc[idx] = False
        else: sm.loc[idx] = False
    return tm, sm


def get_train_test_data(scale_collection,
                        npz_train='./data/Iowa_Lesion.npz',
                        npz_test='./data/ISLES2022_Lesion.npz',
                        cog_train_csv='./cognition/final_train_les_demog25.csv',
                        cog_test_csv='./cognition/final_test_les_demog25.csv'):
    """
    Load, align, and match training/test datasets ensuring demographic balance (age, educ).
    Returns: train_ids, test_ids, train_data, test_data, train_behav, test_behav
    """
    # === load cognition tables ===
    cog = pd.read_csv(cog_train_csv).set_index('study_id')
    cog_test = pd.read_csv(cog_test_csv).set_index('study_id')

    cog_sel = cog[scale_collection].dropna()
    cog_sel_test = cog_test[scale_collection].dropna()
    table_ids = cog_sel.index.astype(int).values
    table_ids_test = cog_sel_test.index.astype(int).values

    # === load lesion npz ===
    npz = np.load(npz_train, allow_pickle=True)
    npz_test = np.load(npz_test, allow_pickle=True)

    def parse_ids(arr):
        if arr.dtype.type is np.bytes_ or arr.dtype.type is np.str_:
            return np.array([int(x.decode() if isinstance(x, bytes) else x) for x in arr], dtype=int)
        return arr.astype(int)

    file_ids = parse_ids(npz['ids'])
    file_ids_test = parse_ids(npz_test['ids'])

    id_to_idx = {int(fid): i for i, fid in enumerate(file_ids)}
    id_to_idx_test = {int(fid): i for i, fid in enumerate(file_ids_test)}

    idx = [id_to_idx[sid] for sid in table_ids if int(sid) in id_to_idx]
    idx_test = [id_to_idx_test[sid] for sid in table_ids_test if int(sid) in id_to_idx_test]

    matched_ids = np.array([sid for sid in table_ids if int(sid) in id_to_idx], dtype=int)
    matched_ids_test = np.array([sid for sid in table_ids_test if int(sid) in id_to_idx_test], dtype=int)

    # === build masks for balancing ===
    complete_mask = pd.Series(False, index=cog.index)
    complete_mask.loc[matched_ids] = True
    complete_mask_test = pd.Series(False, index=cog_test.index)
    complete_mask_test.loc[matched_ids_test] = True

    # === balance demographics ===
    train_mask, test_mask = balance_masks(complete_mask, complete_mask_test, cog, cog_test)

    train_ids = train_mask[train_mask].index.astype(int).to_list()
    test_ids = test_mask[test_mask].index.astype(int).to_list()

    # re-index to ensure alignment with lesion arrays
    train_idx = [id_to_idx[i] for i in train_ids if i in id_to_idx]
    test_idx = [id_to_idx_test[i] for i in test_ids if i in id_to_idx_test]

    # === subset brain features ===
    def subset_data(npz, idx):
        return {
            'sfc_ml_pr': npz['sfc_ml_pr'][idx],
            'sfc_ml_sp': npz['sfc_ml_sp'][idx],
            'sfc_nc_pr': npz['sfc_nc_pr'][idx],
            'sfc_nc_sp': npz['sfc_nc_sp'][idx],
            'sc_ml':     npz['sc_ml'][idx],
            'sc_nc':     npz['sc_nc'][idx],
            'fc':        npz['fc'][idx],
        }

    train_data = subset_data(npz, train_idx)
    test_data  = subset_data(npz_test, test_idx)

    train_behav = cog_sel.loc[train_ids].values
    test_behav  = cog_sel_test.loc[test_ids].values

    return train_ids, test_ids, train_data, test_data, train_behav, test_behav


def get_train_test_data_by_scale(scale_collection,
                        npz_train='./data/Iowa_Lesion.npz',
                        npz_test='./data/ISLES2022_Lesion.npz',
                        cog_train_csv='./cognition/final_train_les_demog25.csv',
                        cog_test_csv='./cognition/final_test_les_demog25.csv'):
    """
    Load and match train/test data per scale independently (no demographic balancing).
    For each scale in `scale_collection`, this returns the maximal set of subjects
    available in that scale (complete-case) present in the corresponding NPZ files.
    """

    # === load cognition tables ===
    cog = pd.read_csv(cog_train_csv).set_index('study_id')
    cog_test = pd.read_csv(cog_test_csv).set_index('study_id')

    # === load npz files ===
    npz = np.load(npz_train, allow_pickle=True)
    npz_test = np.load(npz_test, allow_pickle=True)

    def parse_ids(arr):
        if arr.dtype.type is np.bytes_ or arr.dtype.type is np.str_:
            return np.array([int(x.decode() if isinstance(x, bytes) else x) for x in arr], dtype=int)
        return arr.astype(int)

    file_ids = parse_ids(npz['ids'])
    file_ids_test = parse_ids(npz_test['ids'])
    id_to_idx = {int(fid): i for i, fid in enumerate(file_ids)}
    id_to_idx_test = {int(fid): i for i, fid in enumerate(file_ids_test)}

    # === helper to subset brain features given indices into an npz ===
    def subset_data_from_npz(npz_obj, idx_list):
        """Return dict of feature arrays for selected subjects"""
        if len(idx_list) == 0:
            return {
                'sfc_ml_pr': np.empty((0, ) + npz_obj['sfc_ml_pr'].shape[1:]),
                'sfc_ml_sp': np.empty((0, ) + npz_obj['sfc_ml_sp'].shape[1:]),
                'sfc_nc_pr': np.empty((0, ) + npz_obj['sfc_nc_pr'].shape[1:]),
                'sfc_nc_sp': np.empty((0, ) + npz_obj['sfc_nc_sp'].shape[1:]),
                'sc_ml':     np.empty((0, ) + npz_obj['sc_ml'].shape[1:]),
                'sc_nc':     np.empty((0, ) + npz_obj['sc_nc'].shape[1:]),
                'fc':        np.empty((0, ) + npz_obj['fc'].shape[1:]),
                # === 新增：lesion 特征 ===
                'lesion_mask': np.empty((0, ) + npz_obj['lesion_mask'].shape[1:]),
                'lesion_mask_atlas': np.empty((0, ) + npz_obj['lesion_mask_atlas'].shape[1:]),
            }

        idx_arr = np.array(idx_list, dtype=int)
        return {
            'sfc_ml_pr': npz_obj['sfc_ml_pr'][idx_arr],
            'sfc_ml_sp': npz_obj['sfc_ml_sp'][idx_arr],
            'sfc_nc_pr': npz_obj['sfc_nc_pr'][idx_arr],
            'sfc_nc_sp': npz_obj['sfc_nc_sp'][idx_arr],
            'sc_ml':     npz_obj['sc_ml'][idx_arr],
            'sc_nc':     npz_obj['sc_nc'][idx_arr],
            'fc':        npz_obj['fc'][idx_arr],
            # === 新增：lesion 特征 ===
            'lesion_mask': npz_obj['lesion_mask'][idx_arr],
            'lesion_mask_atlas': npz_obj['lesion_mask_atlas'][idx_arr],
        }

    # === prepare output containers ===
    train_ids_by_scale = {}
    test_ids_by_scale = {}
    train_idx_by_scale = {}
    test_idx_by_scale = {}
    train_data_by_scale = {}
    test_data_by_scale = {}
    train_behav_by_scale = {}
    test_behav_by_scale = {}

    # === iterate scales ===
    for scale in scale_collection:
        # TRAIN side
        if scale not in cog.columns:
            train_ids_by_scale[scale] = []
            train_idx_by_scale[scale] = []
            train_data_by_scale[scale] = subset_data_from_npz(npz, [])
            train_behav_by_scale[scale] = np.empty((0, 1))
        else:
            cog_sel = cog[[scale]].dropna()
            table_ids = cog_sel.index.astype(int).values
            matched_train_ids = [int(sid) for sid in table_ids if int(sid) in id_to_idx]
            train_ids_by_scale[scale] = matched_train_ids
            train_idx = [id_to_idx[sid] for sid in matched_train_ids]
            train_idx_by_scale[scale] = train_idx
            train_data_by_scale[scale] = subset_data_from_npz(npz, train_idx)
            if len(matched_train_ids) > 0:
                train_behav_by_scale[scale] = cog_sel.loc[matched_train_ids].values.reshape(-1, 1)
            else:
                train_behav_by_scale[scale] = np.empty((0, 1))

        # TEST side
        if scale not in cog_test.columns:
            test_ids_by_scale[scale] = []
            test_idx_by_scale[scale] = []
            test_data_by_scale[scale] = subset_data_from_npz(npz_test, [])
            test_behav_by_scale[scale] = np.empty((0, 1))
        else:
            cog_sel_test = cog_test[[scale]].dropna()
            table_ids_test = cog_sel_test.index.astype(int).values
            matched_test_ids = [int(sid) for sid in table_ids_test if int(sid) in id_to_idx_test]
            test_ids_by_scale[scale] = matched_test_ids
            test_idx = [id_to_idx_test[sid] for sid in matched_test_ids]
            test_idx_by_scale[scale] = test_idx
            test_data_by_scale[scale] = subset_data_from_npz(npz_test, test_idx)
            if len(matched_test_ids) > 0:
                test_behav_by_scale[scale] = cog_sel_test.loc[matched_test_ids].values.reshape(-1, 1)
            else:
                test_behav_by_scale[scale] = np.empty((0, 1))

    return (train_ids_by_scale, test_ids_by_scale,
            train_idx_by_scale, test_idx_by_scale,
            train_data_by_scale, test_data_by_scale,
            train_behav_by_scale, test_behav_by_scale)


# def analyze_covariate_influence_in_CCA(Xscore, Yscore, covariates,
#                                r_threshold=0.2, r2_threshold=0.05, verbose=True):
#     """
#     General-purpose function to inspect how much covariates
#     influence canonical scores (or any score vectors).
    
#     Parameters
#     ----------
#     Xscore : array (N, d)
#         Score matrix on the X side.
#     Yscore : array (N, d)
#         Score matrix on the Y side.
#     covariates : DataFrame or array-like (N, K)
#         Covariates to test. Can be:
#         - pandas DataFrame (column names used)
#         - numpy array (column names auto-generated)
#     r_threshold : float
#         |corr| > threshold is considered notable.
#     r2_threshold : float
#         R² > threshold indicates joint covariate contribution.
#     verbose : bool
#         Whether to print detailed results.

#     Returns
#     -------
#     result : dict
#         {
#             'corr': {dim: {cov: {'X': (r,p), 'Y': (r,p)}}},
#             'r2': {dim: {'X': r2_x, 'Y': r2_y}},
#             'recommended_covariates': set([...])
#         }
#     """
#     from scipy.stats import pearsonr
#     import statsmodels.api as sm
#     # Convert covariates to DataFrame if it's not one
#     if isinstance(covariates, pd.DataFrame):
#         cov_df = covariates.copy()
#     else:
#         cov_df = pd.DataFrame(
#             np.asarray(covariates),
#             columns=[f"cov_{i}" for i in range(np.asarray(covariates).shape[1])]
#         )

#     cov_names = list(cov_df.columns)
#     cov_mat = cov_df.values  # (N, K)
    
#     N, d = Xscore.shape
    
#     # Helper for multivariate R²
#     def compute_r2(y, X):
#         X_const = sm.add_constant(X)
#         model = sm.OLS(y, X_const).fit()
#         return model.rsquared

#     corr_dict = {}
#     r2_dict = {}
#     cov_recommend = set()

#     # Loop through selected dimensions
#     for dim in range(d):
#         corr_dict[dim] = {}

#         # ---- 1. simple correlation with each covariate ----
#         for j, cov_name in enumerate(cov_names):
#             cov_vec = cov_mat[:, j]

#             r_x, p_x = pearsonr(Xscore[:, dim], cov_vec)
#             r_y, p_y = pearsonr(Yscore[:, dim], cov_vec)

#             corr_dict[dim][cov_name] = {
#                 'X': (r_x, p_x),
#                 'Y': (r_y, p_y)
#             }

#             # Mark if correlation is notable
#             if abs(r_x) > r_threshold or abs(r_y) > r_threshold:
#                 cov_recommend.add(cov_name)

#         # ---- 2. multivariate R² (joint covariate influence) ----
#         r2_x = compute_r2(Xscore[:, dim], cov_mat)
#         r2_y = compute_r2(Yscore[:, dim], cov_mat)

#         r2_dict[dim] = {'X': r2_x, 'Y': r2_y}

#         if r2_x > r2_threshold or r2_y > r2_threshold:
#             cov_recommend.update(cov_names)

#     # ---- 3. Verbose printout ----
#     if verbose:
#         print("=====================================================")
#         print(" General Covariate Influence Analysis on Scores")
#         print("=====================================================")
        
#         for dim in range(d):
#             print(f"\n---- Dimension {dim} ----")
#             print("Simple correlations (r, p):")
#             for cov_name in cov_names:
#                 r_x, p_x = corr_dict[dim][cov_name]['X']
#                 r_y, p_y = corr_dict[dim][cov_name]['Y']
#                 print(f"  {cov_name}:  X=({r_x:.3f}, p={p_x:.3f})   "
#                       f"Y=({r_y:.3f}, p={p_y:.3f})")

#             print(f"R²:  X={r2_dict[dim]['X']:.3f}   Y={r2_dict[dim]['Y']:.3f}")

#         print("\n>>> Recommended covariates to include in CRM:")
#         print("    ", cov_recommend)
#         print("=====================================================\n")

#     return {
#         'corr': corr_dict,
#         'r2': r2_dict,
#         'recommended_covariates': cov_recommend
#     }

def analyze_covariate_influence_in_CCA(Xscore, Yscore, covariates,
                               p_threshold=0.05, r2_threshold=0.05, verbose=True):
    """
    General-purpose function to inspect how much covariates
    influence canonical scores (or any score vectors).
    
    Parameters
    ----------
    Xscore : array (N, d)
        Score matrix on the X side.
    Yscore : array (N, d)
        Score matrix on the Y side.
    covariates : DataFrame or array-like (N, K)
        Covariates to test.
    p_threshold : float
        p-value < threshold is considered notable (default 0.05).
    r2_threshold : float
        R² > threshold indicates joint covariate contribution.
    verbose : bool
        Whether to print detailed results.

    Returns
    -------
    result : dict
    """
    # Convert covariates to DataFrame if it's not one
    if isinstance(covariates, pd.DataFrame):
        cov_df = covariates.copy()
    else:
        cov_df = pd.DataFrame(
            np.asarray(covariates),
            columns=[f"cov_{i}" for i in range(np.asarray(covariates).shape[1])]
        )

    cov_names = list(cov_df.columns)
    cov_mat = cov_df.values  # (N, K)
    
    N, d = Xscore.shape
    
    # Helper for multivariate R²
    def compute_r2(y, X):
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        return model.rsquared

    corr_dict = {}
    r2_dict = {}
    cov_recommend = set()

    # Loop through selected dimensions
    for dim in range(d):
        corr_dict[dim] = {}

        # ---- 1. simple correlation with each covariate ----
        for j, cov_name in enumerate(cov_names):
            cov_vec = cov_mat[:, j]

            # Calculate Pearson correlation and p-value
            r_x, p_x = pearsonr(Xscore[:, dim], cov_vec)
            r_y, p_y = pearsonr(Yscore[:, dim], cov_vec)

            corr_dict[dim][cov_name] = {
                'X': (r_x, p_x),
                'Y': (r_y, p_y)
            }

            # --- 修改部分：使用 p 值 < 0.05 作为筛选标准 ---
            if p_x < p_threshold or p_y < p_threshold:
                cov_recommend.add(cov_name)

        # ---- 2. multivariate R² (joint covariate influence) ----
        r2_x = compute_r2(Xscore[:, dim], cov_mat)
        r2_y = compute_r2(Yscore[:, dim], cov_mat)

        r2_dict[dim] = {'X': r2_x, 'Y': r2_y}

        if r2_x > r2_threshold or r2_y > r2_threshold:
            cov_recommend.update(cov_names)

    # ---- 3. Verbose printout ----
    if verbose:
        print("=====================================================")
        print(" General Covariate Influence Analysis on Scores")
        print(f" (Significance threshold: p < {p_threshold})")
        print("=====================================================")
        
        for dim in range(d):
            print(f"\n---- Dimension {dim} ----")
            print("Simple correlations (r, p):")
            for cov_name in cov_names:
                r_x, p_x = corr_dict[dim][cov_name]['X']
                r_y, p_y = corr_dict[dim][cov_name]['Y']
                
                # 标记显著的结果
                sig_x = "*" if p_x < p_threshold else " "
                sig_y = "*" if p_y < p_threshold else " "
                
                print(f"  {cov_name}:  X=({r_x:.3f}, p={p_x:.3f}){sig_x}  "
                      f"Y=({r_y:.3f}, p={p_y:.3f}){sig_y}")

            print(f"Multivariate R²:  X={r2_dict[dim]['X']:.3f}   Y={r2_dict[dim]['Y']:.3f}")

        print("\n>>> Recommended covariates to include in CRM:")
        print("    ", cov_recommend)
        print("=====================================================\n")

    return {
        'corr': corr_dict,
        'r2': r2_dict,
        'recommended_covariates': cov_recommend
    }


def deconfound(data, confounders, model=None):
    """
    Regress out confounding variables from the data.
    
    Logic:
    1. If model is None (Training phase): Fit a LinearRegression model (Data ~ Confounds),
       calculate residuals, and return the fitted model.
    2. If model is provided (Validation phase): Use the provided model to predict
       confound effects and calculate residuals. Do NOT refit.

    Formula: Residuals = Data - (beta * Confounds + Intercept)

    Parameters
    ----------
    data : np.ndarray (n_samples, n_features)
        The input data matrix (X or Y) to be deconfounded.
    confounders : np.ndarray (n_samples, n_confounds)
        The confounding variables (e.g., lesion size, age).
    model : sklearn.linear_model.LinearRegression or None
        If None, a new model is fitted to the data.
        If provided, this model is used to predict the confound effects.

    Returns
    -------
    data_resid : np.ndarray
        The deconfounded data (residuals).
    model : sklearn.linear_model.LinearRegression
        The fitted linear regression model (useful for applying to validation sets).
    """
    from sklearn.linear_model import LinearRegression

    # Ensure confounders is 2D array
    if confounders.ndim == 1:
        confounders = confounders.reshape(-1, 1)

    # 1. Fit model if not provided (Training Set)
    if model is None:
        model = LinearRegression(fit_intercept=True)
        model.fit(confounders, data)
    
    # 2. Predict confound effects using the fitted model (Training or Validation)
    # The predict method automatically uses the learned coefficients (beta) and intercept.
    confound_effect = model.predict(confounders)
    
    # 3. Calculate residuals
    data_resid = data - confound_effect
    
    return data_resid, model
