"""Nested cross-validation sensitivity analysis for rCCA model performance.

This script does not replace the original grid-search results. It provides an
independent estimate of held-out canonical correlations when hyperparameters are
selected inside each outer training set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
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


def get_data(
    scale_collection: list[str],
    npz_path: str,
    csv: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    cog_raw = pd.read_csv(csv)
    if "study_id" in cog_raw.columns:
        id_col = "study_id"
    elif "ID" in cog_raw.columns:
        id_col = "ID"
    else:
        raise ValueError("Could not find study_id or ID column in behavior CSV")

    cog = cog_raw.set_index(id_col)
    cog.index = cog.index.astype(int)
    cog = cog[cog["KEEP"] == 1].copy()
    keep_cols = [c for c in scale_collection if c in cog.columns]
    cog_tmp = cog[keep_cols].copy()
    cog_tmp = cog_tmp[~(cog_tmp == 999).any(axis=1)].copy()

    npz = np.load(npz_path, allow_pickle=True)
    file_ids = np.asarray(npz["ids"]).astype(int)
    matched_ids = np.asarray(sorted(set(cog_tmp.index.astype(int)) & set(file_ids)), dtype=int)
    id_to_idx = {sid: i for i, sid in enumerate(file_ids)}
    idx = [id_to_idx[sid] for sid in matched_ids]

    data = {}
    for key in ("sc_nc", "fc"):
        if key not in npz:
            raise KeyError(f"Missing {key} in {npz_path}")
        data[key] = npz[key][idx].astype(np.float32)
    y = cog_tmp.loc[matched_ids].values.astype(np.float32)
    return matched_ids, data, y


def preprocess_feat(a: np.ndarray, mode: str = "norm") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode is None or str(mode).lower() == "none":
        return a, np.zeros((1, a.shape[1])), np.ones((1, a.shape[1]))
    mu = a.mean(axis=0, keepdims=True)
    centered = a - mu
    if str(mode).lower() == "demean":
        return centered, mu, np.ones((1, a.shape[1]))
    if str(mode).lower() == "norm":
        sd = centered.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        return centered / sd, mu, sd
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def make_splits_stratified_by_y(
    ids: np.ndarray,
    y: np.ndarray,
    nfold: int,
    cv_seed: int,
    y_bins: int,
    y_pc: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if y.ndim == 2 and y.shape[1] > 1:
        n_components = min(max(1, y_pc), y.shape[1])
        vec = PCA(n_components=n_components, random_state=cv_seed).fit_transform(y)[:, 0]
    else:
        vec = np.ravel(y)

    quantiles = np.linspace(0, 1, y_bins + 1)
    edges = np.quantile(vec, quantiles)
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-8
    labels = np.digitize(vec, edges[1:-1], right=True).astype(int)
    _, counts = np.unique(labels, return_counts=True)
    if (counts < nfold).any():
        raise ValueError("Some stratification bins have fewer members than nfold")

    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=int(cv_seed))
    return [(tr, va) for tr, va in skf.split(np.zeros(len(ids)), labels)]


def select_features(pack: dict[str, np.ndarray], key: str) -> np.ndarray:
    mapping = {"nc": "sc_nc", "fc": "fc"}
    if key == "nc+fc":
        return np.concatenate([pack["sc_nc"], pack["fc"]], axis=1).astype(np.float32)
    return pack[mapping[key]].astype(np.float32)


def load_confound(csv_path: Path, subject_ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df["study_id"] = df["study_id"].astype(str)
    id_to_lesion = dict(zip(df["study_id"], df["LesionSize"]))
    values = [id_to_lesion.get(str(sid), np.nan) for sid in subject_ids]
    confounds = np.asarray(values, dtype=float).reshape(-1, 1)
    if np.isnan(confounds).any():
        missing = subject_ids[np.isnan(confounds[:, 0])]
        raise ValueError(f"Missing lesion-size confound for IDs: {missing[:10]}")
    return confounds


def regress_confound(
    train_x: np.ndarray,
    train_c: np.ndarray,
    test_x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    model = LinearRegression(fit_intercept=True)
    model.fit(train_c, train_x)
    train_resid = train_x - model.predict(train_c)
    if test_x is None:
        return train_resid, None
    raise ValueError("test confounds must be supplied with test_x")


def regress_confound_train_test(
    train_x: np.ndarray,
    train_c: np.ndarray,
    test_x: np.ndarray,
    test_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = LinearRegression(fit_intercept=True)
    model.fit(train_c, train_x)
    return train_x - model.predict(train_c), test_x - model.predict(test_c)


def fit_transform_outer(
    train_x_raw: np.ndarray,
    train_y_raw: np.ndarray,
    train_c: np.ndarray,
    test_x_raw: np.ndarray,
    test_y_raw: np.ndarray,
    test_c: np.ndarray,
    tau_x: float,
    tau_y: float,
    ndim: int,
    xnorm: str,
    ynorm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x_resid, test_x_resid = regress_confound_train_test(
        train_x_raw, train_c, test_x_raw, test_c
    )
    train_x, mu_x, sd_x = preprocess_feat(train_x_resid, xnorm)
    test_x = (test_x_resid - mu_x) / sd_x
    train_y, mu_y, sd_y = preprocess_feat(train_y_raw, ynorm)
    test_y = (test_y_raw - mu_y) / sd_y

    model = rCCA(latent_dimensions=ndim, c=[tau_x, tau_y], pca=False)
    model.fit((train_x, train_y))
    x_scores, y_scores = model.transform((test_x, test_y))
    x_weights, y_weights = model.weights_
    return x_scores, y_scores, x_weights, y_weights


def inner_cv_score(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    confounds: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    tau_x: float,
    tau_y: float,
    ndim: int,
    eval_dim: int,
    xnorm: str,
    ynorm: str,
) -> float:
    x_score_parts = []
    y_score_parts = []
    y_weight_ref = None

    for tr_idx, va_idx in splits:
        xs, ys, _, y_weights = fit_transform_outer(
            x_raw[tr_idx],
            y_raw[tr_idx],
            confounds[tr_idx],
            x_raw[va_idx],
            y_raw[va_idx],
            confounds[va_idx],
            tau_x,
            tau_y,
            ndim,
            xnorm,
            ynorm,
        )
        if y_weight_ref is None:
            y_weight_ref = y_weights.copy()
            x_score_parts.append(xs)
            y_score_parts.append(ys)
            continue

        corr = np.corrcoef(y_weight_ref.T, y_weights.T)[:ndim, ndim:]
        _, match = linear_sum_assignment(-np.abs(corr))
        signs = np.sign(corr[np.arange(ndim), match])
        signs[signs == 0] = 1
        x_score_parts.append(xs[:, match] * signs)
        y_score_parts.append(ys[:, match] * signs)

    x_all = np.concatenate(x_score_parts, axis=0)
    y_all = np.concatenate(y_score_parts, axis=0)
    rs = []
    for d in range(eval_dim):
        r, _ = pearsonr(x_all[:, d], y_all[:, d])
        rs.append(r)
    rs = np.asarray(rs)
    return float(np.nanmean(rs[:eval_dim]))


def run_nested(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids, data, y = get_data(args.scales, npz_path=args.npz_train, csv=args.cog_train_csv)
    confounds = load_confound(Path(args.lesion_csv), ids)

    tau_grid = [float(v) for v in args.tau_grid]
    outer_splits = make_splits_stratified_by_y(
        np.arange(len(ids)),
        y,
        nfold=args.outer_folds,
        cv_seed=args.seed,
        y_bins=args.y_bins,
        y_pc=args.y_pc,
    )

    fold_rows = []
    summary_rows = []

    for sfc_key in args.sfc:
        x = select_features(data, sfc_key)
        outer_x_parts = []
        outer_y_parts = []
        outer_y_weight_ref = None

        for outer_i, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            x_train = x[train_idx]
            y_train = y[train_idx]
            c_train = confounds[train_idx]

            inner_splits = make_splits_stratified_by_y(
                np.arange(len(train_idx)),
                y_train,
                nfold=args.inner_folds,
                cv_seed=args.seed + outer_i,
                y_bins=args.y_bins,
                y_pc=args.y_pc,
            )

            best = None
            for tau_x in tau_grid:
                for tau_y in tau_grid:
                    score = inner_cv_score(
                        x_train,
                        y_train,
                        c_train,
                        inner_splits,
                        tau_x,
                        tau_y,
                        args.ndim,
                        args.selection_dims,
                        args.xnorm,
                        args.ynorm,
                    )
                    candidate = (score, tau_x, tau_y)
                    if best is None or candidate > best:
                        best = candidate

            assert best is not None
            inner_score, best_tau_x, best_tau_y = best

            xs, ys, _, y_weights = fit_transform_outer(
                x_train,
                y_train,
                c_train,
                x[test_idx],
                y[test_idx],
                confounds[test_idx],
                best_tau_x,
                best_tau_y,
                args.ndim,
                args.xnorm,
                args.ynorm,
            )

            if outer_y_weight_ref is None:
                outer_y_weight_ref = y_weights.copy()
                xs_aligned = xs
                ys_aligned = ys
            else:
                corr = np.corrcoef(outer_y_weight_ref.T, y_weights.T)[:args.ndim, args.ndim:]
                _, match = linear_sum_assignment(-np.abs(corr))
                signs = np.sign(corr[np.arange(args.ndim), match])
                signs[signs == 0] = 1
                xs_aligned = xs[:, match] * signs
                ys_aligned = ys[:, match] * signs

            outer_x_parts.append(xs_aligned)
            outer_y_parts.append(ys_aligned)

            fold_rows.append(
                {
                    "sfc": sfc_key,
                    "outer_fold": outer_i,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "selected_tau_fc": best_tau_x,
                    "selected_tau_behav": best_tau_y,
                    "inner_mean_r_first_dims": inner_score,
                }
            )

        outer_x = np.concatenate(outer_x_parts, axis=0)
        outer_y = np.concatenate(outer_y_parts, axis=0)
        row = {"sfc": sfc_key}
        for d in range(args.eval_dim):
            r, p = pearsonr(outer_x[:, d], outer_y[:, d])
            row[f"nested_r_{d + 1}"] = r
            row[f"nested_p_{d + 1}"] = p
        summary_rows.append(row)

    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-train", default=str(PROJECT_DIR / "data/Iowa_Lesion_MMPI_LNM_features.npz"))
    parser.add_argument("--cog-train-csv", default=str(PROJECT_DIR / "mmpi/Iowa_MMPI2RF_byNandi.csv"))
    parser.add_argument("--lesion-csv", default=str(PROJECT_DIR / "data/Iowa_Lesion_MMPI_LesionSize.csv"))
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "results/nested_cv_sensitivity"))
    parser.add_argument("--sfc", nargs="+", default=["nc", "fc", "nc+fc"], choices=["nc", "fc", "nc+fc"])
    parser.add_argument("--tau-grid", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--ndim", type=int, default=10)
    parser.add_argument("--eval-dim", type=int, default=2)
    parser.add_argument("--selection-dims", type=int, default=2)
    parser.add_argument("--xnorm", default="norm", choices=["none", "demean", "norm"])
    parser.add_argument("--ynorm", default="norm", choices=["none", "demean", "norm"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--y-bins", type=int, default=5)
    parser.add_argument("--y-pc", type=int, default=2)
    parser.add_argument("--scales", nargs="+", default=SCALES_23)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds, summary = run_nested(args)
    folds.to_csv(out_dir / "nested_cv_outer_fold_selections.csv", index=False)
    summary.to_csv(out_dir / "nested_cv_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
