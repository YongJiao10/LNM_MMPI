"""Generate or resume permutation inference for the formal nested rCCA protocol.

The interactive notebooks deliberately load the frozen permutation assets
instead of invoking this script.  Use this command only when a new nested
permutation analysis is explicitly required, and always write it to a new
output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from legacy_cca260.nested_rcca import (
    DEFAULT_TAU_GRID,
    PRIMARY_PARAMS,
    SCALES_23,
    NestedRCCAConfig,
    align_to_reference,
    apply_reference_sign,
    fit_reference,
    fit_train_apply,
    load_data,
    load_lesion_volume,
    make_splits,
    run_nested_rcca,
    score_cv,
    select_features,
    selection_value,
    summarize_scores,
)


WORK: dict[str, Any] = {}


def config_from_args(args: argparse.Namespace, sfc: str) -> NestedRCCAConfig:
    """Build one frozen nested protocol configuration from command-line values."""
    return NestedRCCAConfig(
        sfc=sfc,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        ndim=args.ndim,
        eval_dim=args.eval_dim,
        selection_dims=args.selection_dims,
        selection_metric=args.selection_metric,
        tau_grid=tuple(args.tau_grid),
        seed=args.seed,
        y_bins=args.y_bins,
        y_pc=args.y_pc,
        xnorm=args.xnorm,
        ynorm=args.ynorm,
        scales=tuple(args.scales),
    )


def summarize_nested_permutation(
    sfc: str,
    x: np.ndarray,
    y_permuted: np.ndarray,
    confounds: np.ndarray,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    config: NestedRCCAConfig,
) -> dict[str, float | int | str]:
    """Repeat inner-loop selection and held-out scoring for one Y permutation."""
    primary_tau_x, primary_tau_y = config.resolved_primary_params
    _, reference_y_weights = fit_reference(
        x,
        y_permuted,
        confounds,
        primary_tau_x,
        primary_tau_y,
        config.ndim,
        config.xnorm,
        config.ynorm,
    )

    outer_x_score_parts: list[np.ndarray] = []
    outer_y_score_parts: list[np.ndarray] = []
    selected_tau_x: list[float] = []
    selected_tau_y: list[float] = []

    for outer_fold, (train_index, test_index) in enumerate(outer_splits, start=1):
        inner_splits = make_splits(
            y_permuted[train_index],
            config.inner_folds,
            config.seed + outer_fold,
            config.y_bins,
            config.y_pc,
        )
        best_value = -np.inf
        best_tau_x = np.nan
        best_tau_y = np.nan

        for tau_x in config.tau_grid:
            for tau_y in config.tau_grid:
                candidate_summary = score_cv(
                    x[train_index],
                    y_permuted[train_index],
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
                    best_tau_x = tau_x
                    best_tau_y = tau_y

        x_scores, y_scores, x_weights, y_weights = fit_train_apply(
            x[train_index],
            y_permuted[train_index],
            confounds[train_index],
            x[test_index],
            y_permuted[test_index],
            confounds[test_index],
            best_tau_x,
            best_tau_y,
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
        x_scores, y_scores, _, _ = apply_reference_sign(
            reference_y_weights,
            x_scores,
            y_scores,
            x_weights,
            y_weights,
            config.eval_dim,
        )

        outer_x_score_parts.append(x_scores)
        outer_y_score_parts.append(y_scores)
        selected_tau_x.append(float(best_tau_x))
        selected_tau_y.append(float(best_tau_y))

    summary = summarize_scores(
        np.concatenate(outer_x_score_parts, axis=0),
        np.concatenate(outer_y_score_parts, axis=0),
        [],
        [],
        config.eval_dim,
        include_icc=False,
    )
    return {
        "sfc": sfc,
        "n": int(len(y_permuted)),
        "mean_selected_tau_fc": float(np.mean(selected_tau_x)),
        "mean_selected_tau_behav": float(np.mean(selected_tau_y)),
        **summary,
    }


def init_worker(work: dict[str, Any]) -> None:
    """Install immutable data needed by thread workers."""
    global WORK
    WORK = work


def run_permutation_index(permutation_index: int) -> list[dict[str, float | int | str]]:
    """Run every requested model for one deterministic permutation index."""
    y_permuted = WORK["y"][WORK["permutations"][permutation_index]]
    rows: list[dict[str, float | int | str]] = []
    for sfc in WORK["sfc"]:
        config = WORK["configs"][sfc]
        row = summarize_nested_permutation(
            sfc,
            select_features(WORK["data"], sfc),
            y_permuted,
            WORK["confounds"],
            WORK["outer_splits"],
            config,
        )
        rows.append({"perm_index": int(permutation_index), **row})
    return rows


def load_complete_permutations(path: Path, sfc: list[str]) -> set[int]:
    """Return indices that contain every requested feature set."""
    if not path.exists():
        return set()
    nulls = pd.read_csv(path)
    counts = nulls.groupby("perm_index")["sfc"].nunique()
    return set(counts[counts == len(sfc)].index.astype(int))


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append a completed batch so a long permutation run can be resumed."""
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def read_deduplicated_nulls(path: Path) -> pd.DataFrame:
    """Read and canonicalize resumable permutation rows."""
    nulls = pd.read_csv(path)
    return (
        nulls.sort_values(["perm_index", "sfc"])
        .drop_duplicates(["perm_index", "sfc"], keep="last")
        .reset_index(drop=True)
    )


def add_permutation_p_values(
    summary: pd.DataFrame,
    nulls: pd.DataFrame,
    n_perm: int,
    eval_dim: int,
) -> pd.DataFrame:
    """Add unadjusted right-tail permutation P values to observed summaries."""
    output = summary.copy()
    for dimension in range(1, eval_dim + 1):
        output[f"p_pearson_{dimension}"] = output[f"p_{dimension}"]
        p_values = []
        for row in output.itertuples(index=False):
            sfc_nulls = nulls.loc[nulls["sfc"] == row.sfc, f"r_{dimension}"].to_numpy(
                dtype=float
            )
            if len(sfc_nulls) != n_perm:
                raise RuntimeError(
                    f"{row.sfc} has {len(sfc_nulls)} nulls, expected {n_perm}"
                )
            observed = float(getattr(row, f"r_{dimension}"))
            p_values.append(float((np.sum(sfc_nulls >= observed) + 1) / (n_perm + 1)))
        output[f"p_perm_{dimension}"] = p_values
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-train", default=str(PROJECT_DIR / "data/Iowa_Lesion_MMPI_LNM_features.npz"))
    parser.add_argument("--cog-train-csv", default=str(PROJECT_DIR / "mmpi/Iowa_MMPI2RF_byNandi.csv"))
    parser.add_argument("--lesion-csv", default=str(PROJECT_DIR / "data/Iowa_Lesion_MMPI_LesionSize.csv"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sfc", nargs="+", default=["fc"], choices=sorted(PRIMARY_PARAMS))
    parser.add_argument("--tau-grid", nargs="+", type=float, default=list(DEFAULT_TAU_GRID))
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--ndim", type=int, default=10)
    parser.add_argument("--eval-dim", type=int, default=2)
    parser.add_argument("--selection-dims", type=int, default=2)
    parser.add_argument("--selection-metric", default="mean_r", choices=["mean_r", "min_r", "first_dim"])
    parser.add_argument("--xnorm", default="norm", choices=["none", "demean", "norm"])
    parser.add_argument("--ynorm", default="norm", choices=["none", "demean", "norm"])
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--y-bins", type=int, default=5)
    parser.add_argument("--y-pc", type=int, default=2)
    parser.add_argument("--scales", nargs="+", default=list(SCALES_23))
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--perm-seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially completed output directory instead of overwriting it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_perm < 1 or args.n_jobs < 1 or args.flush_every < 1:
        raise ValueError("n_perm, n_jobs, and flush_every must be positive")

    out_dir = Path(args.out_dir)
    observed_paths = [
        out_dir / "outer_fold_parameters.csv",
        out_dir / "outer_fold_scores.csv",
        out_dir / "summary.csv",
    ]
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{out_dir} is not empty. Use a new directory or pass --resume."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        missing_observed = [str(path) for path in observed_paths if not path.exists()]
        if missing_observed:
            raise FileNotFoundError(
                "--resume requires the observed result files:\\n- "
                + "\\n- ".join(missing_observed)
            )
        summary = pd.read_csv(out_dir / "summary.csv")
    else:
        observed_results = [
            run_nested_rcca(
                config_from_args(args, sfc),
                args.npz_train,
                args.cog_train_csv,
                args.lesion_csv,
            )
            for sfc in args.sfc
        ]
        folds = pd.concat(
            [result.outer_fold_parameters for result in observed_results],
            ignore_index=True,
        )
        scores = pd.concat(
            [result.outer_fold_scores for result in observed_results],
            ignore_index=True,
        )
        summary = pd.concat(
            [result.summary for result in observed_results],
            ignore_index=True,
        )
        folds.to_csv(out_dir / "outer_fold_parameters.csv", index=False)
        scores.to_csv(out_dir / "outer_fold_scores.csv", index=False)
        summary.to_csv(out_dir / "summary.csv", index=False)

    ids, data, y = load_data(tuple(args.scales), args.npz_train, args.cog_train_csv)
    confounds = load_lesion_volume(args.lesion_csv, ids)
    outer_splits = make_splits(y, args.outer_folds, args.seed, args.y_bins, args.y_pc)
    configs = {sfc: config_from_args(args, sfc) for sfc in args.sfc}
    random_state = np.random.RandomState(args.perm_seed)
    permutations = [random_state.permutation(len(ids)) for _ in range(args.n_perm)]

    null_path = out_dir / "permutation_nulls.csv"
    completed = load_complete_permutations(null_path, args.sfc)
    todo = [index for index in range(args.n_perm) if index not in completed]
    init_worker(
        {
            "data": data,
            "y": y,
            "confounds": confounds,
            "outer_splits": outer_splits,
            "configs": configs,
            "sfc": args.sfc,
            "permutations": permutations,
        }
    )

    if args.n_jobs == 1:
        pending_rows: list[dict[str, Any]] = []
        for completed_count, permutation_index in enumerate(todo, start=1):
            pending_rows.extend(run_permutation_index(permutation_index))
            if completed_count % args.flush_every == 0:
                append_rows(null_path, pending_rows)
                pending_rows = []
                print(f"Completed {completed_count}/{len(todo)} pending permutations")
        if pending_rows:
            append_rows(null_path, pending_rows)
    elif todo:
        pending_rows = []
        with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {
                executor.submit(run_permutation_index, index): index
                for index in todo
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                pending_rows.extend(future.result())
                if completed_count % args.flush_every == 0:
                    append_rows(null_path, pending_rows)
                    pending_rows = []
                    print(f"Completed {completed_count}/{len(todo)} pending permutations")
            if pending_rows:
                append_rows(null_path, pending_rows)

    nulls = read_deduplicated_nulls(null_path)
    completed_after = load_complete_permutations(null_path, args.sfc)
    if len(completed_after) != args.n_perm:
        raise RuntimeError(
            f"Expected {args.n_perm} complete permutations, found {len(completed_after)}"
        )
    nulls.to_csv(null_path, index=False)
    summary_with_perm = add_permutation_p_values(
        summary,
        nulls,
        args.n_perm,
        args.eval_dim,
    )
    summary_with_perm.to_csv(out_dir / "summary_with_perm.csv", index=False)

    metadata = {
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "selection_metric": args.selection_metric,
        "selection_dims": args.selection_dims,
        "tau_grid": args.tau_grid,
        "n_perm": args.n_perm,
        "perm_seed": args.perm_seed,
        "seed": args.seed,
        "sfc": args.sfc,
        "permutation_policy": (
            "rows of Y permuted; observed outer folds fixed; "
            "inner tuning repeated per permutation"
        ),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary_with_perm.to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
