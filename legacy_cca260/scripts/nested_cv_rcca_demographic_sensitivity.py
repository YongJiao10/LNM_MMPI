"""Observed nested rCCA sensitivity adjusted for available demographics.

This descriptive sensitivity analysis repeats the observed 10 by 5 nested
procedure without permutation inference. Lesion volume, age at imaging,
registry-recorded gender, and education are removed from brain features within
folds. Age at imaging, registry-recorded gender, and education are removed from
MMPI features within the same folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cca_zoo.linear import rCCA  # noqa: E402


SCALES_23 = [
    "tMLS", "tGIC", "tHPC", "tNUC", "tCOG",
    "tSUI", "tHLP", "tSFD", "tNFC", "tSTW", "tAXY", "tANP", "tBRF", "tMSF",
    "tJCP", "tSUB", "tAGG", "tACT",
    "tFML", "tIPP", "tSAV", "tSHY", "tDSF",
]

PRIMARY_PARAMS = {
    "nc": (0.9, 0.6),
    "fc": (0.9, 0.5),
    "nc+fc": (0.9, 0.4),
}


def compute_icc(weights: np.ndarray) -> float:
    if np.all(weights.var(axis=0, ddof=1) == 0):
        return 1.0
    frame = pd.DataFrame(weights.T).melt(var_name="Raters", value_name="Ratings")
    frame["Items"] = frame.index % weights.shape[1] + 1
    result = pg.intraclass_corr(
        data=frame,
        targets="Items",
        raters="Raters",
        ratings="Ratings",
    )
    return float(result["ICC"].iloc[0])


def load_data(args: argparse.Namespace):
    behavior = pd.read_csv(args.cog_train_csv, encoding="utf-8-sig")
    id_column = "study_id" if "study_id" in behavior.columns else "ID"
    behavior = behavior.set_index(id_column)
    behavior.index = behavior.index.astype(int)
    behavior = behavior.loc[behavior["KEEP"] == 1, SCALES_23]
    behavior = behavior.loc[~(behavior == 999).any(axis=1)]

    feature_pack = np.load(args.npz_train, allow_pickle=True)
    feature_ids = np.asarray(feature_pack["ids"]).astype(int)
    ids = np.asarray(sorted(set(behavior.index) & set(feature_ids)), dtype=int)
    feature_index = {study_id: i for i, study_id in enumerate(feature_ids)}
    row_index = [feature_index[study_id] for study_id in ids]

    data = {
        "sc_nc": feature_pack["sc_nc"][row_index].astype(np.float32),
        "fc": feature_pack["fc"][row_index].astype(np.float32),
    }
    y = behavior.loc[ids].to_numpy(dtype=np.float32)

    lesion = pd.read_csv(args.lesion_csv)
    lesion["study_id"] = lesion["study_id"].astype(int)
    lesion = lesion.set_index("study_id")
    demographics = pd.read_csv(args.demographics_csv)
    demographics["study_id"] = demographics["study_id"].astype(int)
    demographics = demographics.set_index("study_id")

    x_confounds = np.column_stack(
        [
            lesion.loc[ids, "LesionSize"].to_numpy(dtype=float),
            demographics.loc[ids, "scan_age"].to_numpy(dtype=float),
            demographics.loc[ids, "gender"].to_numpy(dtype=float),
            demographics.loc[ids, "educ"].to_numpy(dtype=float),
        ]
    )
    y_confounds = x_confounds[:, 1:].copy()
    if np.isnan(x_confounds).any() or np.isnan(y_confounds).any():
        raise ValueError("Sensitivity covariates contain missing values")
    return ids, data, y, x_confounds, y_confounds


def select_features(data: dict[str, np.ndarray], model: str) -> np.ndarray:
    if model == "nc":
        return data["sc_nc"]
    if model == "fc":
        return data["fc"]
    if model == "nc+fc":
        return np.concatenate([data["sc_nc"], data["fc"]], axis=1)
    raise ValueError(f"Unsupported model: {model}")


def preprocess(values: np.ndarray):
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    sd = centered.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return centered / sd, mean, sd


def residualize_train_test(train, train_covariates, test=None, test_covariates=None):
    model = LinearRegression(fit_intercept=True).fit(train_covariates, train)
    train_residual = train - model.predict(train_covariates)
    if test is None:
        return train_residual, None
    return train_residual, test - model.predict(test_covariates)


def make_splits(y, n_splits, seed, y_bins, y_pc):
    vector = PCA(n_components=y_pc, random_state=seed).fit_transform(y)[:, 0]
    edges = np.quantile(vector, np.linspace(0, 1, y_bins + 1))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-8
    labels = np.digitize(vector, edges[1:-1], right=True)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), labels))


def fit_reference(x, y, x_covariates, y_covariates, tau_x, tau_y, dimensions):
    x_residual, _ = residualize_train_test(x, x_covariates)
    y_residual, _ = residualize_train_test(y, y_covariates)
    x_scaled, _, _ = preprocess(x_residual)
    y_scaled, _, _ = preprocess(y_residual)
    model = rCCA(latent_dimensions=dimensions, c=[tau_x, tau_y], pca=False)
    model.fit((x_scaled, y_scaled))
    return model.weights_


def fit_train_apply(
    x_train,
    y_train,
    x_covariates_train,
    y_covariates_train,
    x_test,
    y_test,
    x_covariates_test,
    y_covariates_test,
    tau_x,
    tau_y,
    dimensions,
):
    x_train_residual, x_test_residual = residualize_train_test(
        x_train, x_covariates_train, x_test, x_covariates_test
    )
    y_train_residual, y_test_residual = residualize_train_test(
        y_train, y_covariates_train, y_test, y_covariates_test
    )
    x_train_scaled, x_mean, x_sd = preprocess(x_train_residual)
    y_train_scaled, y_mean, y_sd = preprocess(y_train_residual)
    x_test_scaled = (x_test_residual - x_mean) / x_sd
    y_test_scaled = (y_test_residual - y_mean) / y_sd

    model = rCCA(latent_dimensions=dimensions, c=[tau_x, tau_y], pca=False)
    model.fit((x_train_scaled, y_train_scaled))
    x_scores, y_scores = model.transform((x_test_scaled, y_test_scaled))
    x_weights, y_weights = model.weights_
    return x_scores, y_scores, x_weights, y_weights


def align(reference_y, x_scores, y_scores, x_weights, y_weights, eval_dim):
    dimensions = reference_y.shape[1]
    correlations = np.corrcoef(reference_y.T, y_weights.T)[:dimensions, dimensions:]
    _, match = linear_sum_assignment(-np.abs(correlations))
    signs = np.sign(correlations[np.arange(dimensions), match])
    signs[signs == 0] = 1
    x_scores = x_scores[:, match] * signs
    y_scores = y_scores[:, match] * signs
    x_weights = x_weights[:, match] * signs
    y_weights = y_weights[:, match] * signs
    for dimension in range(eval_dim):
        if -reference_y[:, dimension].min() > reference_y[:, dimension].max():
            x_scores[:, dimension] *= -1
            y_scores[:, dimension] *= -1
            x_weights[:, dimension] *= -1
            y_weights[:, dimension] *= -1
    return x_scores, y_scores, x_weights, y_weights


def score_inner(x, y, x_covariates, y_covariates, splits, tau_x, tau_y, args):
    _, reference_y = fit_reference(
        x, y, x_covariates, y_covariates, tau_x, tau_y, args.ndim
    )
    x_parts = []
    y_parts = []
    for train_index, test_index in splits:
        result = fit_train_apply(
            x[train_index], y[train_index],
            x_covariates[train_index], y_covariates[train_index],
            x[test_index], y[test_index],
            x_covariates[test_index], y_covariates[test_index],
            tau_x, tau_y, args.ndim,
        )
        x_scores, y_scores, _, _ = align(reference_y, *result, args.eval_dim)
        x_parts.append(x_scores)
        y_parts.append(y_scores)
    x_all = np.concatenate(x_parts)
    y_all = np.concatenate(y_parts)
    correlations = [pearsonr(x_all[:, d], y_all[:, d]).statistic for d in range(args.selection_dims)]
    return float(np.mean(correlations))


def run_model(model_name, ids, x, y, x_covariates, y_covariates, args):
    primary_tau_x, primary_tau_y = PRIMARY_PARAMS[model_name]
    _, reference_y = fit_reference(
        x, y, x_covariates, y_covariates,
        primary_tau_x, primary_tau_y, args.ndim,
    )
    outer_splits = make_splits(y, args.outer_folds, args.seed, args.y_bins, args.y_pc)
    tau_grid = [float(value) for value in args.tau_grid]
    x_parts = []
    y_parts = []
    x_weight_parts = []
    y_weight_parts = []
    fold_rows = []
    score_rows = []

    for fold, (train_index, test_index) in enumerate(outer_splits, start=1):
        inner_splits = make_splits(
            y[train_index], args.inner_folds, args.seed + fold, args.y_bins, args.y_pc
        )
        best = (-np.inf, np.nan, np.nan)
        for tau_x in tau_grid:
            for tau_y in tau_grid:
                value = score_inner(
                    x[train_index], y[train_index],
                    x_covariates[train_index], y_covariates[train_index],
                    inner_splits, tau_x, tau_y, args,
                )
                if value > best[0]:
                    best = (value, tau_x, tau_y)

        result = fit_train_apply(
            x[train_index], y[train_index],
            x_covariates[train_index], y_covariates[train_index],
            x[test_index], y[test_index],
            x_covariates[test_index], y_covariates[test_index],
            best[1], best[2], args.ndim,
        )
        x_scores, y_scores, x_weights, y_weights = align(reference_y, *result, args.eval_dim)
        x_parts.append(x_scores)
        y_parts.append(y_scores)
        x_weight_parts.append(x_weights)
        y_weight_parts.append(y_weights)
        fold_rows.append(
            {
                "sfc": model_name,
                "outer_fold": fold,
                "selected_tau_x": best[1],
                "selected_tau_y": best[2],
                "inner_mean_r": best[0],
            }
        )
        for local_index, study_id in enumerate(ids[test_index]):
            score_rows.append(
                {
                    "sfc": model_name,
                    "outer_fold": fold,
                    "study_id": int(study_id),
                    "x_score_1": x_scores[local_index, 0],
                    "y_score_1": y_scores[local_index, 0],
                    "x_score_2": x_scores[local_index, 1],
                    "y_score_2": y_scores[local_index, 1],
                }
            )

    x_all = np.concatenate(x_parts)
    y_all = np.concatenate(y_parts)
    x_weights = np.stack(x_weight_parts)
    y_weights = np.stack(y_weight_parts)
    summary = {"sfc": model_name, "n": len(ids)}
    for dimension in range(args.eval_dim):
        correlation = pearsonr(x_all[:, dimension], y_all[:, dimension])
        summary[f"r_{dimension + 1}"] = correlation.statistic
        summary[f"p_pearson_{dimension + 1}"] = correlation.pvalue
        summary[f"icc_x_{dimension + 1}"] = compute_icc(x_weights[:, :, dimension])
        summary[f"icc_y_{dimension + 1}"] = compute_icc(y_weights[:, :, dimension])
    return fold_rows, score_rows, summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-train", default=str(PROJECT_DIR / "data/Iowa_Lesion_MMPI_LNM_features.npz"))
    parser.add_argument("--cog-train-csv", default=str(PROJECT_DIR / "mmpi/Iowa_MMPI2RF_byNandi.csv"))
    parser.add_argument("--lesion-csv", default=str(PROJECT_DIR / "data/Iowa_Lesion_MMPI_LesionSize.csv"))
    parser.add_argument("--demographics-csv", default=str(PROJECT_DIR / "data/MMPI_Demographics.csv"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--tau-grid", nargs="+", type=float, default=np.arange(0, 1.01, 0.1).tolist())
    parser.add_argument("--ndim", type=int, default=10)
    parser.add_argument("--eval-dim", type=int, default=2)
    parser.add_argument("--selection-dims", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--y-bins", type=int, default=5)
    parser.add_argument("--y-pc", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    ids, data, y, x_covariates, y_covariates = load_data(args)
    all_folds = []
    all_scores = []
    summaries = []
    for model_name in ("nc", "fc", "nc+fc"):
        folds, scores, summary = run_model(
            model_name,
            ids,
            select_features(data, model_name),
            y,
            x_covariates,
            y_covariates,
            args,
        )
        all_folds.extend(folds)
        all_scores.extend(scores)
        summaries.append(summary)

    pd.DataFrame(all_folds).to_csv(output / "outer_fold_parameters.csv", index=False)
    pd.DataFrame(all_scores).to_csv(output / "outer_fold_scores.csv", index=False)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "summary.csv", index=False)
    metadata = {
        "purpose": "descriptive demographic sensitivity without permutation inference",
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "brain_covariates": ["lesion volume", "age at imaging", "registry-recorded gender", "education"],
        "behavior_covariates": ["age at imaging", "registry-recorded gender", "education"],
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
