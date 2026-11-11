#!/usr/bin/env python3
"""Render the top NET spatial correspondence from targeted receptor decoding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[3]
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from plots import (MULTI_COLOR, NET, PALLETE, PANEL_TITLE_STYLE, SQUARE_PANEL_SIZE,
                   build_brain_network_color_map)  # noqa: E402
from cca401 import preferred


NETWORK_COLOR = build_brain_network_color_map(PALLETE)
MAIN_DIMENSION_COLOR = MULTI_COLOR[:2]
RECEPTOR_DISPLAY = {
    "NET": "norepinephrine transporter (NET)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=preferred.CURRENT / "figure_inputs" / "net_correspondence",
        help="Directory containing the current baseline NET figure inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_DIR / "figs" / "fig6",
        help="Directory for figure outputs.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_best_hits(input_dir: Path) -> pd.DataFrame:
    return pd.read_csv(input_dir / "top_hits" / "selected_top_hits.csv")


def draw_scatter(
    ax,
    pair_df: pd.DataFrame,
    color: str,
    raw_observed_r: float,
    p_spin: float | None = None,
) -> None:
    for net in NET:
        sub = pair_df.loc[pair_df["network"] == net]
        if sub.empty:
            continue
        ax.scatter(
            sub["brain_loading"].to_numpy(dtype=float),
            sub["receptor_value"],
            s=74,
            color=NETWORK_COLOR[net],
            edgecolor=color,
            alpha=0.88,
            linewidth=0.7,
            zorder=3,
        )
    x = pair_df["brain_loading"].to_numpy(dtype=float)
    y = pair_df["receptor_value"].to_numpy(dtype=float)
    coef = np.polyfit(x, y, deg=1)
    xfit = np.linspace(x.min(), x.max(), 200)
    yfit = coef[0] * xfit + coef[1]
    ax.plot(xfit, yfit, color=color, linewidth=2.0)
    ax.set_xlabel("Loading (a.u.)", fontsize=17)
    ax.set_ylabel("PET map value (parcel mean)", fontsize=17)
    ax.set_xlim(-1.05, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=14)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=NETWORK_COLOR[net],
            markeredgecolor="none",
            markersize=9,
            label=net,
            alpha=0.88,
        )
        for net in NET
    ]
    ax.legend(
        handles=legend_handles,
        title="Network",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        frameon=False,
        fontsize=13,
        title_fontsize=14,
        borderaxespad=0.0,
        handletextpad=0.4,
    )
    if p_spin is not None:
        labels = VPacker(
            children=[TextArea(label, textprops={"fontsize": 16})
                      for label in (r"$r$", r"$P_{\mathrm{spin}}$")],
            align="right", pad=0, sep=2,
        )
        values = VPacker(
            children=[TextArea(value, textprops={"fontsize": 16})
                      for value in (f"= {raw_observed_r:.3f}", f"= {p_spin:.4f}")],
            align="left", pad=0, sep=2,
        )
        annotation = AnchoredOffsetbox(
            loc="lower right",
            child=HPacker(children=[labels, values], align="top", pad=0, sep=4),
            bbox_to_anchor=(0.96, 0.02), bbox_transform=ax.transAxes,
            borderpad=0, pad=0.25, prop={"size": 16}, frameon=True,
        )
        annotation.patch.set_boxstyle("round,pad=0,rounding_size=0.25")
        annotation.patch.set_facecolor("white")
        annotation.patch.set_edgecolor(color)
        annotation.patch.set_linewidth(1.2)
        annotation.patch.set_alpha(0.95)
        ax.add_artist(annotation)


def draw_null(ax, null_rs: np.ndarray, raw_observed_r: float, color: str) -> None:
    ax.hist(np.asarray(null_rs, dtype=float), bins=28, color=color, alpha=0.75, edgecolor="none")
    ax.axvline(float(raw_observed_r), color="#222222", linestyle="--", linewidth=1.8)
    ax.set_xlabel(r"Spin-null $r$", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)


def top_hit_title(map_short: str, short_label: str) -> str:
    return f"{short_label} density and {map_short} loadings"


def render_top_hits(input_dir: Path, out_dir: Path) -> None:
    best_hits = load_best_hits(input_dir)
    n_hits = min(2, best_hits.shape[0])
    fig, axes = plt.subplots(n_hits, 1,
                             figsize=(SQUARE_PANEL_SIZE[0], SQUARE_PANEL_SIZE[1] * n_hits))
    if n_hits == 1:
        axes = np.asarray([axes])

    for row_idx, hit in enumerate(best_hits.head(n_hits).itertuples(index=False)):
        color = MAIN_DIMENSION_COLOR[int(hit.dimension) - 1]
        pair_dir = input_dir / "pairwise" / f"{hit.map_id}__{hit.receptor_id}"
        pair_df = pd.read_csv(pair_dir / "parcel_scores.csv")
        null_rs = np.load(pair_dir / "spin_null_r.npy")

        ax = axes[row_idx]
        draw_scatter(ax, pair_df, color, float(hit.observed_r), float(hit.p_spin))
        fig.suptitle(top_hit_title(hit.map_short, hit.short_label), x=0.5,
                     y=0.98, ha="center", **PANEL_TITLE_STYLE)
        ax.set_box_aspect(1)

        # Create inset axes for the null test distribution in the top left
        ax_inset = ax.inset_axes([0.02, 0.78, 0.26, 0.21])
        ax_inset.set_facecolor("none")
        draw_null(ax_inset, null_rs, float(hit.observed_r), color)
        
        # Format the inset null plot: remove the y-axis, enlarge font sizes, remove title
        ax_inset.set_ylabel("")
        ax_inset.get_yaxis().set_visible(False)
        ax_inset.spines["left"].set_visible(False)
        ax_inset.set_xlabel(r"Spin-null $r$", fontsize=13)
        ax_inset.tick_params(axis="x", labelsize=12)
        
    fig.subplots_adjust(left=0.13, right=0.83, bottom=0.14, top=0.85)
    fig.savefig(out_dir / "net_spatial_correspondence.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    render_top_hits(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
