"""Project preprocessing and component alignment for the 4.0.1 nested run.

These data and CV operations reproduce the existing nested_rcca.py and
train_only_nested_core.py protocol without importing the older CCA-Zoo release.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

SCALES_23 = (
    "tMLS", "tGIC", "tHPC", "tNUC", "tCOG", "tSUI", "tHLP", "tSFD", "tNFC",
    "tSTW", "tAXY", "tANP", "tBRF", "tMSF", "tJCP", "tSUB", "tAGG", "tACT",
    "tFML", "tIPP", "tSAV", "tSHY", "tDSF",
)
SCALES_22_WITHOUT_HLP = tuple(scale for scale in SCALES_23 if scale != 'tHLP')
DEFAULT_TAU_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# Fixed labeling anchors only. Final anatomical/molecular coefficients come
# from an independently CV-selected full-data fit in cca401_pipeline.py.
ALIGNMENT_ANCHOR_PARAMS = {"nc": (0.9, 0.6), "fc": (0.9, 0.5), "nc+fc": (0.9, 0.4)}


@dataclass(frozen=True)
class NestedConfig:
    resolved_primary_params: tuple[float, float]
    outer_folds: int = 10
    inner_folds: int = 5
    ndim: int = 10
    eval_dim: int = 2
    selection_dims: int = 2
    selection_metric: str = "mean_r"
    tau_grid: tuple[float, ...] = DEFAULT_TAU_GRID
    seed: int = 1111
    y_bins: int = 5
    y_pc: int = 2
    xnorm: str = "norm"
    ynorm: str = "norm"


def make_config(model, scales):
    if tuple(scales) not in (SCALES_23, SCALES_22_WITHOUT_HLP):
        raise ValueError("Use the approved 23-scale or 22-scale-without-HLP order")
    return NestedConfig(resolved_primary_params=ALIGNMENT_ANCHOR_PARAMS[model])


def load_data(scales, npz_path: str | Path, csv_path: str | Path):
    behavior = pd.read_csv(csv_path)
    id_col = "study_id" if "study_id" in behavior.columns else "ID"
    behavior = behavior.set_index(id_col)
    behavior.index = behavior.index.astype(int)
    behavior = behavior[behavior["KEEP"] == 1].copy()[list(scales)]
    behavior = behavior[~(behavior == 999).any(axis=1)].copy()
    npz = np.load(npz_path, allow_pickle=True)
    file_ids = np.asarray(npz["ids"]).astype(int)
    ids = np.asarray(sorted(set(behavior.index.astype(int)) & set(file_ids)), dtype=int)
    indexes = {study_id: index for index, study_id in enumerate(file_ids)}
    selected = [indexes[study_id] for study_id in ids]
    data = {key: npz[key][selected].astype(np.float32) for key in ("sc_nc", "fc")}
    return ids, data, behavior.loc[ids].values.astype(np.float32)


def select_features(pack, key):
    if key == "nc":
        return pack["sc_nc"].astype(np.float32)
    if key == "fc":
        return pack["fc"].astype(np.float32)
    if key == "nc+fc":
        return np.concatenate([pack["sc_nc"], pack["fc"]], axis=1).astype(np.float32)
    raise ValueError(f"Unknown model: {key}")


def load_total_volumes(lesion_csv, ids):
    table = pd.read_csv(lesion_csv).set_index("study_id")
    return table.loc[ids, "LesionSize"].to_numpy(dtype=float).reshape(-1, 1)


def residualize_train_test(train, train_confounds, test=None, test_confounds=None):
    if train_confounds is None:
        return train.copy(), None if test is None else test.copy()
    model = LinearRegression(fit_intercept=True).fit(train_confounds, train)
    train_residual = train - model.predict(train_confounds)
    if test is None:
        return train_residual, None
    return train_residual, test - model.predict(test_confounds)


def preprocess(values, mode):
    if mode != "norm":
        raise ValueError(f"Unexpected normalization mode: {mode}")
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    sd = centered.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return centered / sd, mean, sd


def make_splits(y, nfold, seed, y_bins, y_pc):
    n_components = min(max(1, y_pc), y.shape[1])
    vector = PCA(n_components=n_components, random_state=seed).fit_transform(y)[:, 0]
    edges = np.quantile(vector, np.linspace(0, 1, y_bins + 1))
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + 1e-8
    labels = np.digitize(vector, edges[1:-1], right=True).astype(int)
    if (np.unique(labels, return_counts=True)[1] < nfold).any():
        raise ValueError("Some stratification bins have fewer members than nfold")
    splitter = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=int(seed))
    return [(train, test) for train, test in splitter.split(np.zeros(len(y)), labels)]


def align_to_reference(reference_y_weights, x_scores, y_scores, x_weights, y_weights):
    ndim = reference_y_weights.shape[1]
    correlations = np.corrcoef(reference_y_weights.T, y_weights.T)[:ndim, ndim:]
    _, components = linear_sum_assignment(-np.abs(correlations))
    signs = np.sign(correlations[np.arange(ndim), components])
    signs[signs == 0] = 1
    return (x_scores[:, components] * signs, y_scores[:, components] * signs,
            x_weights[:, components] * signs, y_weights[:, components] * signs)


def apply_reference_sign(reference_y_weights, x_scores, y_scores, x_weights, y_weights, eval_dim):
    x_scores, y_scores, x_weights, y_weights = [
        value.copy() for value in (x_scores, y_scores, x_weights, y_weights)
    ]
    for dimension in range(min(eval_dim, reference_y_weights.shape[1])):
        if -reference_y_weights[:, dimension].min() > reference_y_weights[:, dimension].max():
            for value in (x_scores, y_scores, x_weights, y_weights):
                value[:, dimension] *= -1
    return x_scores, y_scores, x_weights, y_weights


def harmonize_outer_component_signs(x_score_parts, y_score_parts, x_weight_parts,
                                    y_weight_parts, eval_dim):
    signs = np.ones((len(y_weight_parts), eval_dim), dtype=float)
    anchor = y_weight_parts[0]
    for fold in range(1, len(y_weight_parts)):
        for dimension in range(eval_dim):
            correlation = float(np.corrcoef(anchor[:, dimension], y_weight_parts[fold][:, dimension])[0, 1])
            if not np.isfinite(correlation):
                raise ValueError(f"Cannot harmonize fold {fold + 1}, component {dimension + 1}")
            if correlation < 0:
                signs[fold, dimension] = -1.0
    for fold, fold_signs in enumerate(signs):
        for parts in (x_score_parts, y_score_parts, x_weight_parts, y_weight_parts):
            parts[fold][:, :eval_dim] *= fold_signs
    return signs


from utils import compute_icc


def selection_value(score_row, metric, selection_dims):
    if metric != "mean_r":
        raise ValueError(f"Unexpected selection criterion: {metric}")
    return float(np.nanmean([score_row[f"r_{dimension + 1}"] for dimension in range(selection_dims)]))
