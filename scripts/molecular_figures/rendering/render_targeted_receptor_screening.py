#!/usr/bin/env python3
"""Render the current targeted 10-map receptor overview."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

MPLCONFIG_DIR = Path("/private/tmp/lnm_cca_matplotlib")
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIR))
from cca401 import preferred
from plots import PANEL_TITLE_STYLE, SQUARE_PANEL_SIZE

FIT_DIR, _ = preferred.selected_fit("nc+fc")
SOURCE_DIR = preferred.MOLECULAR_PRECISION / FIT_DIR.relative_to(preferred.CURRENT / "fits")

RECEPTOR_ORDER = ["D1", "D2/3", "5-HTT", "5-HT1A", "5-HT2A", "NET", "GABA-A", "mGluR5", "MOR", "VAChT"]
SYSTEM_COLORS = {
    "D1": "#D08C2F",
    "D2/3": "#C17A21",
    "5-HTT": "#3B75AF",
    "5-HT1A": "#4B8CCB",
    "5-HT2A": "#2E6FAE",
    "NET": "#008F8C",
    "GABA-A": "#7B5EA7",
    "mGluR5": "#C75146",
    "MOR": "#5D6D7E",
    "VAChT": "#6A994E",
}
MAP_COLORS = {"CD1-FDC": "#459395", "CD2-FDC": "#866F85"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help="Directory containing targeted receptor-decoding outputs.",
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


def load_summary(source_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(source_dir / "pet.csv").rename(columns={
        "receptor": "short_label", "r": "observed_r",
        "q_fdr_within_cd_10": "q_within_map", "q_fdr_20": "q_all_tests",
    })
    df["map_short"] = df["map_id"].map({"combined_cd1_fdc": "CD1-FDC", "combined_cd2_fdc": "CD2-FDC"})
    df["fdr_hit_within_map"] = df["q_within_map"] < 0.05
    df["fdr_hit_all_tests"] = df["q_all_tests"] < 0.05
    df["receptor_rank"] = df["short_label"].map({name: idx for idx, name in enumerate(RECEPTOR_ORDER)})
    df = df.sort_values(["map_short", "receptor_rank"]).reset_index(drop=True)
    return df


def fdr_label(row: pd.Series) -> str:
    return ""


def draw_lollipop(ax: plt.Axes, data: pd.DataFrame, map_short: str) -> None:
    sub = data.loc[data["map_short"] == map_short].copy()
    y_lookup = {name: len(RECEPTOR_ORDER) - 1 - idx for idx, name in enumerate(RECEPTOR_ORDER)}
    for _, row in sub.iterrows():
        y = y_lookup[row["short_label"]]
        r = float(row["observed_r"])
        color = SYSTEM_COLORS[row["short_label"]]
        ax.hlines(y, 0, r, color=color, linewidth=2.7, alpha=0.82, zorder=2)
        is_hit = bool(row["fdr_hit_within_map"])
        ax.scatter(
            r,
            y,
            s=1.6 * (92 + 250 * abs(r)),
            facecolor=color if is_hit else "white",
            edgecolor=color,
            linewidth=2.5 if is_hit else 2.0,
            zorder=4,
        )
        if is_hit:
            stars = "*"
            ax.text(
                r,
                y + 0.26,
                stars,
                color=color,
                ha="center",
                va="bottom",
                fontsize=14.5,
                fontweight="bold",
                zorder=5,
            )
        label = fdr_label(row)
        if label:
            ax.text(
                r + (-0.035 if r < 0 else 0.035),
                y + 0.02,
                label,
                ha="right" if r < 0 else "left",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="#111111",
            )
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlim(-0.72, 0.64)
    ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    ax.set_ylim(-0.75, len(RECEPTOR_ORDER) - 0.25)
    ax.set_title(map_short, fontsize=20, fontweight="bold", color=MAP_COLORS[map_short], pad=6)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def set_receptor_labels(ax: plt.Axes) -> None:
    y_positions = [len(RECEPTOR_ORDER) - 1 - idx for idx, _ in enumerate(RECEPTOR_ORDER)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(RECEPTOR_ORDER, fontsize=16)
    for tick in ax.get_yticklabels():
        label = tick.get_text()
        tick.set_color(SYSTEM_COLORS[label])
        tick.set_fontweight("normal")


def add_net_spin_inset(ax: plt.Axes, data: pd.DataFrame) -> None:
    pass


def render_overview(data: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    fig = plt.figure(figsize=SQUARE_PANEL_SIZE)
    ax_cd1 = fig.add_subplot(111)

    draw_lollipop(ax_cd1, data, "CD1-FDC")
    set_receptor_labels(ax_cd1)
    ax_cd1.set_ylabel("")
    ax_cd1.set_xlabel("")
    add_net_spin_inset(ax_cd1, data)
    fig.suptitle("Receptor and transporter correspondence", x=0.5, y=0.98,
                 ha="center", **PANEL_TITLE_STYLE)
    fig.supxlabel("Spatial correlation (r)", fontsize=17, x=0.61, y=0.055)
    fig.subplots_adjust(left=0.25, right=0.97, bottom=0.13, top=0.84)
    fig.savefig(output_dir / "targeted_receptor_screening.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_summary(args.source_dir)
    render_overview(data, args.output_dir)


if __name__ == "__main__":
    main()
