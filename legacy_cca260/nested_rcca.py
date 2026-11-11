"""Historical2.6.0 manuscript-baseline implementation, not the current runner.

Current4.0.1 selection, full-data refits and molecular inputs are implemented
in cca401_pipeline.py. This module is retained only for the author's original
baseline and its numerical comparisons; its reference weights are not current
final-model maps.

This module is intentionally specific to the lesion and MMPI analyses.  It
implements the frozen primary protocol used for the manuscript nested results:
feature residualization within every training fold, stratified outer and inner
cross-validation, regularization selection within each outer training set, and
reference-based component alignment.

It has no output-writing side effects.  Callers receive all quantities needed
for notebook summaries, validation, and visualizations in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pingouin as pg
from cca_zoo.linear import rCCA
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold


SCALES_23 = (
    "tMLS", "tGIC", "tHPC", "tNUC", "tCOG",
    "tSUI", "tHLP", "tSFD", "tNFC", "tSTW", "tAXY", "tANP", "tBRF", "tMSF",
    "tJCP", "tSUB", "tAGG", "tACT",
    "tFML", "tIPP", "tSAV", "tSHY", "tDSF",
)

PRIMARY_PARAMS: dict[str, tuple[float, float]] = {
    "nc": (0.9, 0.6),
    "fc": (0.9, 0.5),
    "nc+fc": (0.9, 0.4),
}

# Keep the literal values from the formal command-line implementation.  Using
# np.arange here would make values such as 0.3 marginally different in binary
# floating point and can prevent exact score replay.
DEFAULT_TAU_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
FeatureSet = Literal["nc", "fc", "nc+fc"]
SelectionMetric = Literal["mean_r", "min_r", "first_dim"]
Normalization = Literal["none", "demean", "norm"]


@dataclass(frozen=True)
class NestedRCCAConfig:
    """Configuration for one observed nested rCCA model."""

    sfc: FeatureSet = "fc"
    outer_folds: int = 10
    inner_folds: int = 5
    ndim: int = 10
    eval_dim: int = 2
    selection_dims: int = 2
    selection_metric: SelectionMetric = "mean_r"
    tau_grid: tuple[float, ...] = DEFAULT_TAU_GRID
    seed: int = 1111
    y_bins: int = 5
    y_pc: int = 2
    xnorm: Normalization = "norm"
    ynorm: Normalization = "norm"
    scales: tuple[str, ...] = SCALES_23
    primary_params: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.sfc not in PRIMARY_PARAMS:
            raise ValueError(f"Unsupported feature set: {self.sfc}")
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("outer_folds and inner_folds must each be at least 2")
        if self.ndim < 1 or self.eval_dim < 1 or self.selection_dims < 1:
            raise ValueError("ndim, eval_dim, and selection_dims must be positive")
        if self.eval_dim > self.ndim:
            raise ValueError("eval_dim cannot exceed ndim")
        if self.selection_dims > self.eval_dim:
            raise ValueError("selection_dims cannot exceed eval_dim")
        if not self.tau_grid:
            raise ValueError("tau_grid cannot be empty")
        if any(not 0.0 <= value <= 1.0 for value in self.tau_grid):
            raise ValueError("All rCCA regularization values must be in [0, 1]")
        if self.xnorm not in {"none", "demean", "norm"}:
            raise ValueError(f"Unsupported X normalization: {self.xnorm}")
        if self.ynorm not in {"none", "demean", "norm"}:
            raise ValueError(f"Unsupported Y normalization: {self.ynorm}")
        if self.primary_params is not None and len(self.primary_params) != 2:
            raise ValueError("primary_params must contain the X and behavioral tau values")

    @property
    def resolved_primary_params(self) -> tuple[float, float]:
        """Primary full-sample parameters used only for reference alignment."""
        return self.primary_params or PRIMARY_PARAMS[self.sfc]


@dataclass
class NestedRCCAResult:
    """In-memory outputs from the observed nested rCCA analysis."""

    ids: np.ndarray
    outer_test_indices: np.ndarray
    reference_x_weights: np.ndarray
    reference_y_weights: np.ndarray
    outer_x_weights: np.ndarray
    outer_y_weights: np.ndarray
    heldout_x_scores: np.ndarray
    heldout_y_scores: np.ndarray
    outer_fold_parameters: pd.DataFrame
    outer_fold_scores: pd.DataFrame
    summary: pd.DataFrame


def load_data(
    scale_collection: tuple[str, ...],
    npz_path: str | Path,
    csv_path: str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Load the exact participant subset used by the formal nested analysis."""
    behavior_raw = pd.read_csv(csv_path)
    if "study_id" in behavior_raw.columns:
        id_col = "study_id"
    elif "ID" in behavior_raw.columns:
        id_col = "ID"
    else:
        raise ValueError("Could not find study_id or ID column in behavior CSV")

    behavior = behavior_raw.set_index(id_col)
    behavior.index = behavior.index.astype(int)
    behavior = behavior[behavior["KEEP"] == 1].copy()

    missing_scales = [scale for scale in scale_collection if scale not in behavior.columns]
    if missing_scales:
        raise ValueError(f"Missing MMPI scale columns: {missing_scales}")

    behavior = behavior[list(scale_collection)].copy()
    behavior = behavior[~(behavior == 999).any(axis=1)].copy()

    npz = np.load(npz_path, allow_pickle=True)
    file_ids = np.asarray(npz["ids"]).astype(int)
    matched_ids = np.asarray(sorted(set(behavior.index.astype(int)) & set(file_ids)), dtype=int)
    id_to_idx = {study_id: index for index, study_id in enumerate(file_ids)}
    matched_indices = [id_to_idx[study_id] for study_id in matched_ids]

    data = {
        "sc_nc": npz["sc_nc"][matched_indices].astype(np.float32),
        "fc": npz["fc"][matched_indices].astype(np.float32),
    }
    y = behavior.loc[matched_ids].values.astype(np.float32)
    return matched_ids, data, y


def select_features(pack: dict[str, np.ndarray], key: FeatureSet) -> np.ndarray:
    """Return the prescribed structural, functional, or combined feature matrix."""
    if key == "nc":
        return pack["sc_nc"].astype(np.float32)
    if key == "fc":
        return pack["fc"].astype(np.float32)
    if key == "nc+fc":
        return np.concatenate([pack["sc_nc"], pack["fc"]], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported feature key: {key}")


def load_lesion_volume(csv_path: str | Path, subject_ids: np.ndarray) -> np.ndarray:
    """Load lesion volume and enforce one available value per modeled participant."""
    table = pd.read_csv(csv_path)
    if "study_id" not in table.columns or "LesionSize" not in table.columns:
        raise ValueError("Lesion CSV must contain study_id and LesionSize")

    table["study_id"] = table["study_id"].astype(str)
    id_to_lesion = dict(zip(table["study_id"], table["LesionSize"]))
    values = [id_to_lesion.get(str(study_id), np.nan) for study_id in subject_ids]
    confounds = np.asarray(values, dtype=float).reshape(-1, 1)
    if np.isnan(confounds).any():
        missing = subject_ids[np.isnan(confounds[:, 0])]
        raise ValueError(f"Missing lesion-volume confound for IDs: {missing[:10]}")
    return confounds


def preprocess(
    values: np.ndarray,
    mode: Normalization,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit featurewise preprocessing on one training matrix."""
    if mode == "none":
        return values, np.zeros((1, values.shape[1])), np.ones((1, values.shape[1]))

    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    if mode == "demean":
        return centered, mean, np.ones((1, values.shape[1]))
    if mode == "norm":
        standard_deviation = centered.std(axis=0, keepdims=True)
        standard_deviation[standard_deviation == 0] = 1.0
        return centered / standard_deviation, mean, standard_deviation
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def make_splits(
    y: np.ndarray,
    nfold: int,
    seed: int,
    y_bins: int,
    y_pc: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create the frozen PC1-quantile stratified folds used by the formal run."""
    if y.ndim == 2 and y.shape[1] > 1:
        n_components = min(max(1, y_pc), y.shape[1])
        stratification_vector = PCA(
            n_components=n_components,
            random_state=seed,
        ).fit_transform(y)[:, 0]
    else:
        stratification_vector = np.ravel(y)

    edges = np.quantile(stratification_vector, np.linspace(0, 1, y_bins + 1))
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + 1e-8
    labels = np.digitize(stratification_vector, edges[1:-1], right=True).astype(int)

    _, counts = np.unique(labels, return_counts=True)
    if (counts < nfold).any():
        raise ValueError("Some stratification bins have fewer members than nfold")

    splitter = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=int(seed))
    return [(train_index, test_index) for train_index, test_index in splitter.split(
        np.zeros(len(y)),
        labels,
    )]


def residualize_train_test(
    x_train_raw: np.ndarray,
    c_train: np.ndarray,
    x_test_raw: np.ndarray | None = None,
    c_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit lesion-volume regression on training data and apply it to held-out data."""
    model = LinearRegression(fit_intercept=True)
    model.fit(c_train, x_train_raw)
    x_train = x_train_raw - model.predict(c_train)
    if x_test_raw is None:
        return x_train, None
    if c_test is None:
        raise ValueError("c_test is required when x_test_raw is supplied")
    return x_train, x_test_raw - model.predict(c_test)


def fit_reference(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    confounds: np.ndarray,
    tau_x: float,
    tau_y: float,
    ndim: int,
    xnorm: Normalization,
    ynorm: Normalization,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the full-sample reference model used for component labeling."""
    x_residualized, _ = residualize_train_test(x_raw, confounds)
    x, _, _ = preprocess(x_residualized, xnorm)
    y, _, _ = preprocess(y_raw, ynorm)
    model = rCCA(latent_dimensions=ndim, c=[tau_x, tau_y], pca=False)
    model.fit((x, y))
    return model.weights_


def fit_train_apply(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    c_train: np.ndarray,
    x_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
    c_test: np.ndarray,
    tau_x: float,
    tau_y: float,
    ndim: int,
    xnorm: Normalization,
    ynorm: Normalization,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one training fold and score its held-out participants."""
    x_train_residualized, x_test_residualized = residualize_train_test(
        x_train_raw,
        c_train,
        x_test_raw,
        c_test,
    )
    x_train, mean_x, standard_deviation_x = preprocess(x_train_residualized, xnorm)
    x_test = (x_test_residualized - mean_x) / standard_deviation_x
    y_train, mean_y, standard_deviation_y = preprocess(y_train_raw, ynorm)
    y_test = (y_test_raw - mean_y) / standard_deviation_y

    model = rCCA(latent_dimensions=ndim, c=[tau_x, tau_y], pca=False)
    model.fit((x_train, y_train))
    x_scores, y_scores = model.transform((x_test, y_test))
    x_weights, y_weights = model.weights_
    return x_scores, y_scores, x_weights, y_weights


def align_to_reference(
    reference_y_weights: np.ndarray,
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    x_weights: np.ndarray,
    y_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match each fold's components and signs to the full-sample reference."""
    ndim = reference_y_weights.shape[1]
    correlations = np.corrcoef(reference_y_weights.T, y_weights.T)[:ndim, ndim:]
    _, matched_components = linear_sum_assignment(-np.abs(correlations))
    signs = np.sign(correlations[np.arange(ndim), matched_components])
    signs[signs == 0] = 1
    return (
        x_scores[:, matched_components] * signs,
        y_scores[:, matched_components] * signs,
        x_weights[:, matched_components] * signs,
        y_weights[:, matched_components] * signs,
    )


def apply_reference_sign(
    reference_y_weights: np.ndarray,
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    x_weights: np.ndarray,
    y_weights: np.ndarray,
    eval_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the formal behavioral-loading sign convention to evaluated dimensions."""
    x_scores = x_scores.copy()
    y_scores = y_scores.copy()
    x_weights = x_weights.copy()
    y_weights = y_weights.copy()

    for dimension in range(min(eval_dim, reference_y_weights.shape[1])):
        if -reference_y_weights[:, dimension].min() > reference_y_weights[:, dimension].max():
            x_scores[:, dimension] *= -1
            y_scores[:, dimension] *= -1
            x_weights[:, dimension] *= -1
            y_weights[:, dimension] *= -1
    return x_scores, y_scores, x_weights, y_weights


def align_outer_outputs_to_reference_for_display(
    reference_y_weights: np.ndarray,
    outer_x_weights: np.ndarray,
    outer_y_weights: np.ndarray,
    heldout_x_scores: np.ndarray,
    heldout_y_scores: np.ndarray,
    eval_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return display copies whose outer components share the reference polarity.

    The formal nested result remains untouched because its held-out score signs
    are part of the frozen authority replay.  Canonical-component signs are
    arbitrary, however, and a reference solution can retain the opposite sign
    after the formal outer-score convention is applied.  This helper resolves
    that presentation-only ambiguity by using the mean outer behavioral
    loading for each evaluated component as the comparison target.  A detected
    reversal is applied jointly to the paired outer X/Y weights and held-out
    X/Y scores, preserving all correlation, P-value, ICC, FDR, and tuning
    quantities.

    Returns
    -------
    outer_x_weights_display, outer_y_weights_display,
    heldout_x_scores_display, heldout_y_scores_display, outer_signs
        ``outer_signs`` contains one multiplier per fitted dimension.  Values
        outside ``eval_dim`` are always 1 and therefore remain formal-output
        copies.
    """
    if reference_y_weights.ndim != 2:
        raise ValueError("reference_y_weights must be 2-dimensional")
    if outer_x_weights.ndim != 3 or outer_y_weights.ndim != 3:
        raise ValueError("outer weight arrays must be 3-dimensional")
    if heldout_x_scores.ndim != 2 or heldout_y_scores.ndim != 2:
        raise ValueError("held-out score arrays must be 2-dimensional")
    if outer_y_weights.shape[1:] != reference_y_weights.shape:
        raise ValueError(
            "outer_y_weights feature/dimension shape must match reference_y_weights"
        )
    if outer_x_weights.shape[0] != outer_y_weights.shape[0]:
        raise ValueError("outer X and Y weights must have the same fold count")
    if heldout_x_scores.shape != heldout_y_scores.shape:
        raise ValueError("held-out X and Y scores must have the same shape")
    if heldout_x_scores.shape[1] != reference_y_weights.shape[1]:
        raise ValueError("held-out score dimensions must match reference weights")
    if not 1 <= eval_dim <= reference_y_weights.shape[1]:
        raise ValueError("eval_dim must be within the fitted component range")

    display_x_weights = outer_x_weights.copy()
    display_y_weights = outer_y_weights.copy()
    display_x_scores = heldout_x_scores.copy()
    display_y_scores = heldout_y_scores.copy()
    outer_signs = np.ones(reference_y_weights.shape[1], dtype=float)

    for dimension in range(eval_dim):
        reference = reference_y_weights[:, dimension]
        outer_mean = outer_y_weights[:, :, dimension].mean(axis=0)
        if np.isclose(reference.std(), 0) or np.isclose(outer_mean.std(), 0):
            raise ValueError(
                f"Cannot determine display polarity for component {dimension + 1}: "
                "constant behavioral weights"
            )
        correlation = float(np.corrcoef(reference, outer_mean)[0, 1])
        if not np.isfinite(correlation):
            raise ValueError(
                f"Cannot determine display polarity for component {dimension + 1}"
            )
        if correlation < 0:
            outer_signs[dimension] = -1.0

    display_x_weights *= outer_signs[np.newaxis, np.newaxis, :]
    display_y_weights *= outer_signs[np.newaxis, np.newaxis, :]
    display_x_scores *= outer_signs[np.newaxis, :]
    display_y_scores *= outer_signs[np.newaxis, :]
    return (
        display_x_weights,
        display_y_weights,
        display_x_scores,
        display_y_scores,
        outer_signs,
    )


from loading_icc import compute_icc


def summarize_scores(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    x_weights: list[np.ndarray],
    y_weights: list[np.ndarray],
    eval_dim: int,
    include_icc: bool = True,
) -> dict[str, float]:
    """Calculate held-out score correlations and, when requested, foldwise ICCs."""
    summary: dict[str, float] = {}
    stacked_x_weights = np.stack(x_weights) if include_icc else None
    stacked_y_weights = np.stack(y_weights) if include_icc else None

    for dimension in range(eval_dim):
        correlation, p_value = pearsonr(x_scores[:, dimension], y_scores[:, dimension])
        summary[f"r_{dimension + 1}"] = float(correlation)
        summary[f"p_{dimension + 1}"] = float(p_value)
        if include_icc:
            summary[f"icc_x_{dimension + 1}"] = float(
                compute_icc(stacked_x_weights[:, :, dimension])
            )
            summary[f"icc_y_{dimension + 1}"] = float(
                compute_icc(stacked_y_weights[:, :, dimension])
            )
    return summary


def selection_value(
    score_row: dict[str, float],
    metric: SelectionMetric,
    selection_dims: int,
) -> float:
    """Reduce the prespecified inner-CV dimensions to the tuning objective."""
    correlations = np.asarray(
        [score_row[f"r_{dimension + 1}"] for dimension in range(selection_dims)],
        dtype=float,
    )
    if metric == "mean_r":
        return float(np.nanmean(correlations))
    if metric == "min_r":
        return float(np.nanmin(correlations))
    if metric == "first_dim":
        return float(correlations[0])
    raise ValueError(f"Unknown selection metric: {metric}")


def score_cv(
    x: np.ndarray,
    y: np.ndarray,
    confounds: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    tau_x: float,
    tau_y: float,
    config: NestedRCCAConfig,
) -> dict[str, float]:
    """Score one tau pair in an inner cross-validation loop."""
    _, reference_y_weights = fit_reference(
        x,
        y,
        confounds,
        tau_x,
        tau_y,
        config.ndim,
        config.xnorm,
        config.ynorm,
    )
    x_score_parts: list[np.ndarray] = []
    y_score_parts: list[np.ndarray] = []
    x_weight_parts: list[np.ndarray] = []
    y_weight_parts: list[np.ndarray] = []

    for train_index, valid_index in splits:
        x_scores, y_scores, x_weights, y_weights = fit_train_apply(
            x[train_index],
            y[train_index],
            confounds[train_index],
            x[valid_index],
            y[valid_index],
            confounds[valid_index],
            tau_x,
            tau_y,
            config.ndim,
            config.xnorm,
            config.ynorm,
        )
        x_scores, y_scores, x_weights, y_weights = align_to_reference(
            reference_y_weights,
            x_scores,
            y_scores,
            x_weights,
            y_weights,
        )
        x_scores, y_scores, x_weights, y_weights = apply_reference_sign(
            reference_y_weights,
            x_scores,
            y_scores,
            x_weights,
            y_weights,
            config.eval_dim,
        )
        x_score_parts.append(x_scores)
        y_score_parts.append(y_scores)
        x_weight_parts.append(x_weights)
        y_weight_parts.append(y_weights)

    return summarize_scores(
        np.concatenate(x_score_parts, axis=0),
        np.concatenate(y_score_parts, axis=0),
        x_weight_parts,
        y_weight_parts,
        config.eval_dim,
        include_icc=False,
    )


def run_nested_rcca(
    config: NestedRCCAConfig,
    npz_train: str | Path,
    cog_train_csv: str | Path,
    lesion_csv: str | Path,
) -> NestedRCCAResult:
    """Run observed nested rCCA without writing any files."""
    ids, data, y = load_data(config.scales, npz_train, cog_train_csv)
    confounds = load_lesion_volume(lesion_csv, ids)
    x = select_features(data, config.sfc)

    primary_tau_x, primary_tau_y = config.resolved_primary_params
    reference_x_weights, reference_y_weights = fit_reference(
        x,
        y,
        confounds,
        primary_tau_x,
        primary_tau_y,
        config.ndim,
        config.xnorm,
        config.ynorm,
    )
    outer_splits = make_splits(
        y,
        config.outer_folds,
        config.seed,
        config.y_bins,
        config.y_pc,
    )

    fold_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    outer_x_score_parts: list[np.ndarray] = []
    outer_y_score_parts: list[np.ndarray] = []
    outer_x_weight_parts: list[np.ndarray] = []
    outer_y_weight_parts: list[np.ndarray] = []
    outer_test_index_parts: list[np.ndarray] = []

    for outer_fold, (train_index, test_index) in enumerate(outer_splits, start=1):
        inner_splits = make_splits(
            y[train_index],
            config.inner_folds,
            config.seed + outer_fold,
            config.y_bins,
            config.y_pc,
        )
        best_value = -np.inf
        selected_tau_x = np.nan
        selected_tau_y = np.nan
        inner_selection_score = np.nan

        for tau_x in config.tau_grid:
            for tau_y in config.tau_grid:
                candidate_summary = score_cv(
                    x[train_index],
                    y[train_index],
                    confounds[train_index],
                    inner_splits,
                    tau_x,
                    tau_y,
                    config,
                )
                candidate_value = selection_value(
                    candidate_summary,
                    config.selection_metric,
                    config.selection_dims,
                )
                if candidate_value > best_value:
                    best_value = candidate_value
                    selected_tau_x = tau_x
                    selected_tau_y = tau_y
                    inner_selection_score = candidate_value

        x_scores, y_scores, x_weights, y_weights = fit_train_apply(
            x[train_index],
            y[train_index],
            confounds[train_index],
            x[test_index],
            y[test_index],
            confounds[test_index],
            selected_tau_x,
            selected_tau_y,
            config.ndim,
            config.xnorm,
            config.ynorm,
        )
        x_scores, y_scores, x_weights, y_weights = align_to_reference(
            reference_y_weights,
            x_scores,
            y_scores,
            x_weights,
            y_weights,
        )
        x_scores, y_scores, x_weights, y_weights = apply_reference_sign(
            reference_y_weights,
            x_scores,
            y_scores,
            x_weights,
            y_weights,
            config.eval_dim,
        )

        outer_x_score_parts.append(x_scores)
        outer_y_score_parts.append(y_scores)
        outer_x_weight_parts.append(x_weights)
        outer_y_weight_parts.append(y_weights)
        outer_test_index_parts.append(test_index)

        fold_rows.append(
            {
                "mode": "nested_tuned",
                "sfc": config.sfc,
                "outer_fold": outer_fold,
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "selected_tau_fc": float(selected_tau_x),
                "selected_tau_behav": float(selected_tau_y),
                "inner_selection_score": float(inner_selection_score),
                "alignment_reference": "full_sample_primary_params_for_labeling_only",
            }
        )
        for local_index, study_id in enumerate(ids[test_index]):
            row: dict[str, object] = {
                "mode": "nested_tuned",
                "sfc": config.sfc,
                "outer_fold": outer_fold,
                "study_id": int(study_id),
            }
            for dimension in range(config.eval_dim):
                row[f"x_score_{dimension + 1}"] = float(x_scores[local_index, dimension])
                row[f"y_score_{dimension + 1}"] = float(y_scores[local_index, dimension])
            score_rows.append(row)

    heldout_x_scores = np.concatenate(outer_x_score_parts, axis=0)
    heldout_y_scores = np.concatenate(outer_y_score_parts, axis=0)
    outer_x_weights = np.stack(outer_x_weight_parts)
    outer_y_weights = np.stack(outer_y_weight_parts)
    outer_test_indices = np.concatenate(outer_test_index_parts)

    summary_row: dict[str, object] = {
        "mode": "nested_tuned",
        "sfc": config.sfc,
        "n": int(len(ids)),
        "outer_folds": int(config.outer_folds),
        "inner_folds": int(config.inner_folds),
        "selection_metric": config.selection_metric,
        "selection_dims": int(config.selection_dims),
        "primary_tau_fc": float(primary_tau_x),
        "primary_tau_behav": float(primary_tau_y),
        "mean_selected_tau_fc": float(np.mean([row["selected_tau_fc"] for row in fold_rows])),
        "mean_selected_tau_behav": float(
            np.mean([row["selected_tau_behav"] for row in fold_rows])
        ),
        "alignment_reference": "full_sample_primary_params_for_labeling_only",
    }
    summary_row.update(
        summarize_scores(
            heldout_x_scores,
            heldout_y_scores,
            outer_x_weight_parts,
            outer_y_weight_parts,
            config.eval_dim,
        )
    )

    return NestedRCCAResult(
        ids=ids,
        outer_test_indices=outer_test_indices,
        reference_x_weights=reference_x_weights,
        reference_y_weights=reference_y_weights,
        outer_x_weights=outer_x_weights,
        outer_y_weights=outer_y_weights,
        heldout_x_scores=heldout_x_scores,
        heldout_y_scores=heldout_y_scores,
        outer_fold_parameters=pd.DataFrame(fold_rows),
        outer_fold_scores=pd.DataFrame(score_rows),
        summary=pd.DataFrame([summary_row]),
    )
