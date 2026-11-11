#!/usr/bin/env python3
"""Render ranked canonical CD2 FDC stable gene weights."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors


CODE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIR))
from cca401 import preferred

FIT_DIR, _ = preferred.selected_fit("nc+fc")
INPUT = preferred.MOLECULAR_PRECISION / FIT_DIR.relative_to(preferred.CURRENT / "fits") / "combined_cd2_fdc" / "gene_weights.csv"
POS_COLOR = "#B53F3F"
NEG_COLOR = "#305C8A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=INPUT,
        help="Current baseline CD2 FDC gene-weight table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_DIR / "figs" / "supp_gene_stability",
        help="Directory for figure outputs.",
    )
    return parser.parse_args()


def blend_with_white(color: str, strength: float) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(1.0 - strength * (1.0 - rgb))


def main() -> None:
    args = parse_args()
    output = args.output_dir / "cd2_fdc_stable_gene_weights"
    data = pd.read_csv(args.input_file).rename(columns={"bootstrap_ratio": "bootstrap_ratio_display"}).sort_values(
        "bootstrap_ratio_display", ascending=False
    )
    stable = data.loc[data["q_boot_fdr"] < 0.05].copy()
    positive = stable.loc[stable["bootstrap_ratio_display"] > 0]
    negative = stable.loc[stable["bootstrap_ratio_display"] < 0]

    rows: list[tuple[str, float] | None] = []
    rows.extend(
        (str(row["gene"]), float(row["bootstrap_ratio_display"]))
        for _, row in positive.head(4).iterrows()
    )
    rows.append(None)
    rows.append(
        (
            str(positive.tail(1).iloc[0]["gene"]),
            float(positive.tail(1).iloc[0]["bootstrap_ratio_display"]),
        )
    )
    rows.append(
        (
            str(negative.head(1).iloc[0]["gene"]),
            float(negative.head(1).iloc[0]["bootstrap_ratio_display"]),
        )
    )
    rows.append(None)
    rows.extend(
        (str(row["gene"]), float(row["bootstrap_ratio_display"]))
        for _, row in negative.tail(4).iterrows()
    )

    plt.rcParams.update({"pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    t = ax.text(
         0.5,
         0.98,
         "Ranked CD2 FDC gene weights",
         ha="center",
         va="bottom",
         fontsize=13.5,
         fontweight="bold",
    )
    ax.text(
         0.5,
         0.935,
         "Social-Behavioral Dysregulation",
         ha="center",
         va="bottom",
         fontsize=10.5,
         color="#5E6870",
    )
    
    # Dynamically align table lines and columns with the title width
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_t = t.get_window_extent(renderer=renderer)
    x0_data = ax.transData.inverted().transform((bbox_t.x0, bbox_t.y0))[0]
    x1_data = ax.transData.inverted().transform((bbox_t.x1, bbox_t.y1))[0]
    
    ax.plot([x0_data, x1_data], [0.895, 0.895], color="black", linewidth=1.0)
    ax.text(x0_data + 0.04, 0.85, "Gene", fontsize=12.0, fontweight="bold")
    ax.text(x1_data - 0.04, 0.85, "Bootstrap ratio", ha="right", fontsize=12.0, fontweight="bold")
    ax.plot([x0_data, x1_data], [0.815, 0.815], color="black", linewidth=0.85)

    y_positions = np.linspace(0.75, 0.125, len(rows))
    pos_idx = 0
    neg_idx = 0
    for row, y in zip(rows, y_positions):
        if row is None:
            ax.text(x0_data + 0.08, y, r"$\vdots$", ha="center", va="center", fontsize=14)
            ax.text(x1_data - 0.08, y, r"$\vdots$", ha="center", va="center", fontsize=14)
            continue
        gene, value = row
        base = POS_COLOR if value > 0 else NEG_COLOR
        
        # Rank-based linear gradient: 1.0 down to 0.40 for positive, 0.40 up to 1.0 for negative
        if value > 0:
            strength = 1.0 - 0.60 * (pos_idx / 4)
            pos_idx += 1
        else:
            strength = 0.40 + 0.60 * (neg_idx / 4)
            neg_idx += 1
            
        color = blend_with_white(base, strength)
        ax.text(x0_data + 0.04, y, gene, fontsize=12.0, fontstyle="italic", color=color, va="center")
        ax.text(x1_data - 0.04, y, f"{value:.2f}", ha="right", fontsize=12.0, color=color, va="center")

    ax.plot([x0_data, x1_data], [0.08, 0.08], color="black", linewidth=0.85)
    ax.text(
        0.5,
        0.025,
        f"{len(positive):,} display positive and {len(negative):,} display negative stable genes",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#5E6870",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
