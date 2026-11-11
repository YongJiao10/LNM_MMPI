#!/usr/bin/env python3
"""Render canonicalized AHBA spatial correspondence across combined-model maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


CODE_DIR = Path(__file__).resolve().parents[3]
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from plots import (  # noqa: E402
    MULTI_COLOR,
    PAIR_COLOR,
    PALLETE,
    PANEL_TITLE_STYLE,
    SQUARE_PANEL_SIZE,
    build_brain_network_color_map,
)
from cca401 import preferred


NETWORK_ORDER = ["VN", "SMN", "DAN", "VAN", "LN", "FPCN", "DMN"]
NETWORK_COLOR_MAP = build_brain_network_color_map(PALLETE)
MAIN_DIMENSION_COLOR = MULTI_COLOR[:2]
MODALITY_COLOR = {
    "sdc": PAIR_COLOR[0][0],
    "fdc": PAIR_COLOR[0][1],
}
fit_directory, _ = preferred.selected_fit("nc+fc")
RESULTS = preferred.MOLECULAR_PRECISION / fit_directory.relative_to(preferred.CURRENT / "fits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render canonicalized AHBA summary figure.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RESULTS,
        help="Directory containing the selected baseline molecular results.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="AHBA decoding of cortical loadings",
        help="Figure title.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_DIR / "figs" / "fig6",
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--scatter-dir",
        type=Path,
        default=preferred.CURRENT / "figure_inputs" / "ahba_scatter",
        help="Directory containing the prepared baseline parcel scores.",
    )
    args = parser.parse_args()

    args.output_stem = args.output_dir / "ahba_spatial_correspondence"

    return args


def load_inputs(input_dir: Path, scatter_dir: Path):
    map_ids = ["combined_cd1_sdc", "combined_cd1_fdc",
               "combined_cd2_sdc", "combined_cd2_fdc"]
    summary = pd.read_csv(input_dir / "pls.csv").set_index("map_id").loc[map_ids].reset_index()
    summary = summary.rename(columns={"r": "observed_r_display"})
    summary["dimension"] = [1, 1, 2, 2]
    summary["modality"] = ["sdc", "fdc", "sdc", "fdc"]
    summary["map_short"] = ["CD1-SDC", "CD1-FDC", "CD2-SDC", "CD2-FDC"]
    scatter_tables = {map_id: pd.read_csv(scatter_dir / f"{map_id}.csv") for map_id in map_ids}
    null_arrays = {map_id: np.load(input_dir / map_id / "spin_null_abs_r.npy") for map_id in map_ids}
    return summary, scatter_tables, null_arrays


def draw_scatter_panel(
    ax,
    df: pd.DataFrame,
    row: pd.Series,
    null_values: np.ndarray | None = None,
) -> None:
    modality = str(row["modality"]).lower()
    dim_idx = int(row["dimension"]) - 1
    edge_color = MODALITY_COLOR[modality]
    line_color = MAIN_DIMENSION_COLOR[dim_idx]

    sns.regplot(
        data=df,
        x="weight_raw",
        y="pls1_gene_score_map_display",
        scatter=False,
        ci=None,
        color=line_color,
        line_kws={"linewidth": 2.2, "alpha": 0.95},
        ax=ax,
    )

    for net in NETWORK_ORDER:
        sub = df.loc[df["network"] == net]
        if sub.empty:
            continue
        ax.scatter(
            sub["weight_raw"],
            sub["pls1_gene_score_map_display"],
            s=46,
            marker="o",
            color=NETWORK_COLOR_MAP[net],
            edgecolor=edge_color,
            linewidth=0.8,
            alpha=0.88,
            zorder=3,
        )

    ax.set_title(
        row["map_short"],
        fontsize=20,
        fontweight="bold",
        color=line_color,
        pad=8,
    )
    ax.text(
        0.04,
        0.96,
        f"r = {row['observed_r_display']:.3f}\n$P_{{\\mathrm{{spin}}}}$ = {row['p_spin']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12.5,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": line_color,
            "linewidth": 1.2,
            "alpha": 0.95,
        },
    )
    ax.set_xlabel("Loading (a.u.)", fontsize=14)
    ax.set_ylabel("PLS1 gene-expression score (a.u.)", fontsize=14)
    ax.set_xlim(-1.05, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(labelsize=13)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)

    if null_values is not None:
        ax_inset = ax.inset_axes([0.70, 0.14, 0.23, 0.23])
        ax_inset.set_facecolor("none")
        draw_null_panel(ax_inset, null_values, row, compact=True)


def draw_null_panel(ax, null_values: np.ndarray, row: pd.Series, compact: bool = False) -> None:
    modality = str(row["modality"]).lower()
    dim_idx = int(row["dimension"]) - 1
    fill_color = MODALITY_COLOR[modality]
    line_color = MAIN_DIMENSION_COLOR[dim_idx]

    ax.hist(
        null_values,
        bins=24,
        color=fill_color,
        alpha=0.55,
        edgecolor="none",
    )
    ax.axvline(row["observed_r_display"], color=line_color, linewidth=2.5, linestyle="--")
    if compact:
        ax.set_xlabel(r"Spin-null $r$", fontsize=7.4, labelpad=0.5)
        ax.set_ylabel("")
        ax.get_yaxis().set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=10.0, pad=0)
    else:
        ax.set_xlabel(r"Spin-null $r$", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.tick_params(labelsize=13)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



def build_network_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor=NETWORK_COLOR_MAP[net],
            markeredgecolor="none",
            label=net,
        )
        for net in NETWORK_ORDER
    ]


def save_main_summary_figure(
    output_stem: Path,
    title: str,
    canonical_summary: pd.DataFrame,
    scatter_tables: dict[str, pd.DataFrame],
    null_arrays: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=SQUARE_PANEL_SIZE)
    axes_flat = axes.flatten()
    for idx, ((_, row), ax) in enumerate(zip(canonical_summary.iterrows(), axes_flat)):
        map_id = str(row["map_id"])
        draw_scatter_panel(ax, scatter_tables[map_id], row, null_arrays[map_id])

    fig.legend(
        handles=build_network_handles(),
        title="Network",
        frameon=False,
        fontsize=14,
        title_fontsize=15,
        loc="center left",
        bbox_to_anchor=(0.84, 0.50),
        borderaxespad=0.0,
        handletextpad=0.4,
    )

    fig.suptitle(title, x=0.5, y=0.98, ha="center", **PANEL_TITLE_STYLE)
    fig.subplots_adjust(left=0.10, right=0.82, bottom=0.10, top=0.87,
                        wspace=0.44, hspace=0.46)
    fig.savefig(f"{output_stem}.pdf", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_stem = args.output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "savefig.transparent": False,
        }
    )

    canonical_summary, scatter_tables, null_arrays = load_inputs(input_dir, args.scatter_dir)

    save_main_summary_figure(output_stem, args.title, canonical_summary, scatter_tables, null_arrays)

    print(f"Saved: {output_stem}.pdf")


if __name__ == "__main__":
    main()
