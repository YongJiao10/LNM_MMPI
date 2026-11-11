"""Descriptive demographic adjustment of fixed primary held-out rCCA scores.

This analysis does not refit rCCA or redefine the canonical dimensions. For
each primary model and dimension, it residualizes the held-out brain and
behavior canonical scores for age at imaging, registry-recorded gender, and
education, then correlates the two residual series.

The output is descriptive. It does not provide permutation-based inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


PROJECT_DIR = Path(__file__).resolve().parents[3]


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(covariates)), covariates])
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def analyze_model(
    model: str,
    score_path: Path,
    demographics: pd.DataFrame,
) -> list[dict[str, float | int | str]]:
    scores = pd.read_csv(score_path)
    required = {
        "study_id",
        "x_score_1",
        "y_score_1",
        "x_score_2",
        "y_score_2",
    }
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"{score_path} is missing columns: {sorted(missing)}")
    if scores["study_id"].duplicated().any():
        raise ValueError(f"{score_path} contains duplicate study_id values")

    merged = scores.merge(
        demographics,
        on="study_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(scores):
        raise ValueError(
            f"Demographic merge retained {len(merged)} of {len(scores)} rows for {model}"
        )

    covariate_names = ["scan_age", "gender", "educ"]
    if merged[covariate_names].isna().any().any():
        raise ValueError(f"Missing demographic covariates for {model}")
    covariates = merged[covariate_names].to_numpy(dtype=float)

    rows = []
    for dimension in (1, 2):
        x_scores = merged[f"x_score_{dimension}"].to_numpy(dtype=float)
        y_scores = merged[f"y_score_{dimension}"].to_numpy(dtype=float)
        raw_r = float(pearsonr(x_scores, y_scores).statistic)
        adjusted_r = float(
            pearsonr(
                residualize(x_scores, covariates),
                residualize(y_scores, covariates),
            ).statistic
        )
        rows.append(
            {
                "model": model,
                "dimension": f"CD{dimension}",
                "n": len(merged),
                "primary_held_out_r": raw_r,
                "demographic_adjusted_partial_r": adjusted_r,
                "change_in_r": adjusted_r - raw_r,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demographics",
        type=Path,
        default=PROJECT_DIR / "LNM_MMPI_code/data/MMPI_Demographics.csv",
    )
    parser.add_argument(
        "--structural-scores",
        type=Path,
        default=PROJECT_DIR
        / "review_prep/nested_results_review_0706/"
        "seed1111_10x5_sdc_1000perm_0708/outer_fold_scores.csv",
    )
    parser.add_argument(
        "--functional-scores",
        type=Path,
        default=PROJECT_DIR
        / "review_prep/nested_results_review_0706/"
        "seed1111_10x5_fdc_1000perm_0709/outer_fold_scores.csv",
    )
    parser.add_argument(
        "--combined-scores",
        type=Path,
        default=PROJECT_DIR
        / "review_prep/nested_results_review_0706/"
        "seed1111_primary_score_demographic_adjustment_0729/"
        "primary_inputs/combined_outer_fold_scores.csv",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demographics = pd.read_csv(args.demographics)
    required_demographics = {"study_id", "scan_age", "gender", "educ"}
    missing = required_demographics.difference(demographics.columns)
    if missing:
        raise ValueError(
            f"{args.demographics} is missing columns: {sorted(missing)}"
        )
    if demographics["study_id"].duplicated().any():
        raise ValueError("Demographic file contains duplicate study_id values")

    score_paths = {
        "Structural": args.structural_scores,
        "Functional": args.functional_scores,
        "Combined": args.combined_scores,
    }
    rows = []
    for model, score_path in score_paths.items():
        rows.extend(analyze_model(model, score_path, demographics))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    metadata = {
        "status": "internal reserve analysis; not reported in the manuscript",
        "purpose": (
            "descriptive demographic adjustment of fixed primary held-out "
            "canonical scores"
        ),
        "covariates": [
            "age at imaging",
            "registry-recorded gender",
            "education",
        ],
        "rcca_refitted": False,
        "canonical_dimensions_redefined": False,
        "permutation_inference": False,
        "interpretive_scope": (
            "tests whether linear demographic associations explain the "
            "correlation between fixed primary held-out canonical scores"
        ),
        "inputs": {key: str(value) for key, value in score_paths.items()},
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
