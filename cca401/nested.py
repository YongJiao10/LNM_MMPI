"""Training-only preprocessing and matched nested evaluation, native 4.0.1."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from cca_zoo.linear import RidgeCCA
from scipy.stats import pearsonr
from .nested_utils import *
CODE_DIR = Path(__file__).resolve().parents[1]

@dataclass
class PreparedSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray | None
    y_test: np.ndarray | None

def prepare_split(x, y, volume, train, test, policy, config):
    """Estimate feature adjustment and feature standardization on training data."""
    c_train = None if policy == "none" else volume[train]
    c_test = None if policy == "none" or test is None else volume[test]
    x_test = None if test is None else x[test]
    y_test = None if test is None else y[test]
    x_train, x_test = residualize_train_test(x[train], c_train, x_test, c_test)
    if policy == "both":
        y_train, y_test = residualize_train_test(y[train], c_train, y_test, c_test)
    else:
        y_train = y[train].copy()
        y_test = None if y_test is None else y_test.copy()
    x_train, x_mean, x_sd = preprocess(x_train, config.xnorm)
    y_train, y_mean, y_sd = preprocess(y_train, config.ynorm)
    if test is not None:
        x_test = (x_test - x_mean) / x_sd
        y_test = (y_test - y_mean) / y_sd
    return PreparedSplit(x_train, y_train, x_test, y_test)

def fit_reference(prepared, tau_x, tau_y, config):
    model = RidgeCCA(n_components=config.ndim, shrinkage=[tau_x, tau_y])
    model.fit([prepared.x_train.astype(np.float64), prepared.y_train.astype(np.float64)])
    return model.weights_[1]

def fit_scores(prepared, reference_y, tau_x, tau_y, config, score_scaling):
    """Fit and align test projections, optionally scaling from training scores."""
    model = RidgeCCA(n_components=config.ndim, shrinkage=[tau_x, tau_y])
    x_train = prepared.x_train.astype(np.float64)
    y_train = prepared.y_train.astype(np.float64)
    model.fit([x_train, y_train])
    x_score, y_score = model.transform([
        prepared.x_test.astype(np.float64), prepared.y_test.astype(np.float64)
    ])
    x_weight, y_weight = model.weights_
    aligned = align_to_reference(reference_y, x_score, y_score, x_weight, y_weight)
    x_score, y_score, x_weight, y_weight = apply_reference_sign(
        reference_y, *aligned, config.eval_dim
    )
    x_train_score = (x_train - model.means_[0]) @ x_weight
    y_train_score = (y_train - model.means_[1]) @ y_weight
    x_mean, x_sd = x_train_score.mean(axis=0), x_train_score.std(axis=0)
    y_mean, y_sd = y_train_score.mean(axis=0), y_train_score.std(axis=0)
    if not np.all(np.isfinite(x_sd)) or not np.all(np.isfinite(y_sd)):
        raise ValueError("Non-finite training score standard deviation")
    if np.any(x_sd <= 0) or np.any(y_sd <= 0):
        raise ValueError("Zero training score standard deviation")
    if score_scaling == "training":
        x_score = (x_score - x_mean) / x_sd
        y_score = (y_score - y_mean) / y_sd
    return (
        x_score,
        y_score,
        x_weight,
        y_weight,
        x_mean,
        x_sd,
        y_mean,
        y_sd,
    )

def score_inner(reference_data, inner_data, tau_x, tau_y, config, score_scaling):
    """Preserve the original pooled two-dimension inner selection criterion."""
    reference_y = fit_reference(reference_data, tau_x, tau_y, config)
    pieces = [fit_scores(part, reference_y, tau_x, tau_y, config, score_scaling) for part in inner_data]
    x = np.concatenate([part[0] for part in pieces])
    y = np.concatenate([part[1] for part in pieces])
    return {
        f"r_{dimension + 1}": float(pearsonr(x[:, dimension], y[:, dimension]).statistic)
        for dimension in range(config.eval_dim)
    }

def calculate_fold(ids, x, y, volume, outer, fold, policy, config, score_scaling):
    train, test = outer[fold - 1]
    xt, yt, vt = x[train], y[train], volume[train]
    inner = make_splits(
        yt, config.inner_folds, config.seed + fold, config.y_bins, config.y_pc
    )
    reference_data = prepare_split(
        xt, yt, vt, np.arange(len(train)), None, policy, config
    )
    inner_data = [prepare_split(xt, yt, vt, tr, va, policy, config) for tr, va in inner]
    best_value, best_tau_x, best_tau_y = -np.inf, np.nan, np.nan
    for tau_x in config.tau_grid:
        for tau_y in config.tau_grid:
            candidate = score_inner(reference_data, inner_data, tau_x, tau_y, config, score_scaling)
            value = selection_value(candidate, config.selection_metric, config.selection_dims)
            if value > best_value:
                best_value, best_tau_x, best_tau_y = value, tau_x, tau_y
    outer_data = prepare_split(x, y, volume, train, test, policy, config)
    reference_x, reference_y = config.resolved_primary_params
    reference_weights = fit_reference(reference_data, reference_x, reference_y, config)
    result = fit_scores(outer_data, reference_weights, best_tau_x, best_tau_y, config, score_scaling)
    return {
        "ids": ids[test],
        "x_score": result[0],
        "y_score": result[1],
        "x_weight": result[2],
        "y_weight": result[3],
        "x_train_mean": result[4],
        "x_train_sd": result[5],
        "y_train_mean": result[6],
        "y_train_sd": result[7],
        "tau_x": np.array(best_tau_x),
        "tau_y": np.array(best_tau_y),
        "inner_selection_score": np.array(best_value),
    }
