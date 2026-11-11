#!/usr/bin/env python3
"""Render pathway enrichment for the CD1-FDC gene sets."""

import argparse
from pathlib import Path
import re
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[3]
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
from cca401 import preferred
from plots import PANEL_TITLE_STYLE, SQUARE_PANEL_SIZE

fit_directory, _ = preferred.selected_fit("nc+fc")
RESULTS = preferred.MOLECULAR_PRECISION / fit_directory.relative_to(preferred.CURRENT / "fits")
POS_COLOR = "#C8971C"
NEG_COLOR = "#835BD6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Directory containing finalized AHBA outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_DIR / "figs" / "fig6",
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=None,
        help="Optional directory for the selected enrichment-term table.",
    )
    return parser.parse_args()


def clean_term(term: str) -> str:
    term = re.sub(r"\s*\([^()]*GO:\d+[^()]*\)\s*$", "", str(term))
    term = re.sub(r"\s*\(R-HSA-\d+\)\s*$", "", term)
    term = re.sub(r"\s+R-HSA-\d+\s*$", "", term)
    return re.sub(r"\s+", " ", term.replace("Homo sapiens", "").strip())


def term_prefix(gene_set: str) -> str:
    if gene_set.startswith("GO_"):
        return "GO:"
    if gene_set.startswith("KEGG"):
        return "KEGG:"
    if gene_set.startswith("Reactome"):
        return "Reactome:"
    return ""


def wrap_term_label(label: str) -> str:
    lines = textwrap.wrap(
        label, width=40, break_long_words=False, break_on_hyphens=False
    )
    for index in range(len(lines) - 1, 0, -1):
        words = lines[index - 1].split()
        while len(words) > 1:
            left = " ".join(words[:-1])
            right = f"{words[-1]} {lines[index]}"
            current_difference = abs(len(lines[index - 1]) - len(lines[index]))
            if len(right) > 40 or abs(len(left) - len(right)) >= current_difference:
                break
            lines[index - 1], lines[index] = left, right
            words.pop()
    return "\n".join(lines)


def load_terms(results_dir: Path, map_id: str, display_sign: str) -> pd.DataFrame:
    raw_sign = display_sign
    sign_name = {"pos": "positive", "neg": "negative"}[raw_sign]
    data = pd.read_csv(results_dir / map_id / f"{sign_name}_enrichment.csv")
    data = data.rename(columns={"library": "Gene_set", "term": "Term",
                                "q": "Adjusted P-value", "p": "P-value"})
    data["Overlap"] = data["overlap"].astype(str) + "/" + data["term_size"].astype(str)
    data = data.loc[data["Adjusted P-value"] < 0.05].copy()
    data = data.sort_values(["Adjusted P-value", "P-value"])

    selected = []
    per_database: dict[str, int] = {}
    for _, row in data.iterrows():
        gene_set = str(row["Gene_set"])
        if per_database.get(gene_set, 0) >= 2:
            continue
        selected.append(row)
        per_database[gene_set] = per_database.get(gene_set, 0) + 1
        if len(selected) >= 6:
            break

    result = pd.DataFrame(selected).copy()
    result["Term_clean"] = result["Term"].map(clean_term)
    result["neg_log10_q"] = -np.log10(result["Adjusted P-value"].astype(float))
    result["count"] = result["Overlap"].astype(str).str.split("/").str[0].astype(int)
    result["map_id"] = map_id
    result["display_sign"] = display_sign
    result["raw_sign"] = raw_sign
    return result


def main() -> None:
    args = parse_args()
    output = args.output_dir / "fdc_pathway_enrichment"
    output.parent.mkdir(parents=True, exist_ok=True)
    group_order = [
        ("combined_cd1_fdc", "pos"),
        ("combined_cd1_fdc", "neg"),
    ]
    parts = [load_terms(args.results_dir, map_id, sign) for map_id, sign in group_order]
    data = pd.concat(parts, ignore_index=True)
    if args.records_dir is not None:
        args.records_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.records_dir / "fdc_pathway_enrichment_terms.csv", index=False)

    ordered_parts = []
    boundaries = []
    cursor = 0
    for map_id, sign in group_order:
        part = data.loc[
            (data["map_id"] == map_id) & (data["display_sign"] == sign)
        ].sort_values("neg_log10_q", ascending=False)
        start = cursor
        cursor += len(part)
        boundaries.append((start, cursor - 1))
        ordered_parts.append(part)
    plot_data = pd.concat(ordered_parts, ignore_index=True)
    plt.rcParams.update({"pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=SQUARE_PANEL_SIZE)
    labels = [
        wrap_term_label(f"{term_prefix(gene_set)}{term}")
        for gene_set, term in zip(plot_data["Gene_set"], plot_data["Term_clean"])
    ]
    # Give wrapped terms enough room without spreading every row equally.
    row_heights = np.array([label.count("\n") + 1.55 for label in labels])
    row_bottoms = row_heights.sum() - np.cumsum(row_heights)
    plot_data["y"] = row_bottoms + row_heights / 2
    sizes = 34 + 4.6 * plot_data["count"].to_numpy(dtype=float)
    sign_colors = {"pos": POS_COLOR, "neg": NEG_COLOR}
    map_markers = {"combined_cd1_fdc": "o"}

    # Draw bubbles
    for (map_id, sign), part in plot_data.groupby(
        ["map_id", "display_sign"], sort=False
    ):
        indices = part.index.to_numpy()
        ax.scatter(
            part["neg_log10_q"],
            part["y"],
            s=sizes[indices],
            marker=map_markers[map_id],
            color=sign_colors[sign],
            edgecolor="white",
            linewidth=0.75,
            alpha=0.9,
            zorder=3,
        )

    # Formatting axes and labels
    ax.set_yticks(plot_data["y"])
    ax.set_yticklabels(labels, fontsize=12.5, linespacing=0.95)
    for label, sign in zip(ax.get_yticklabels(), plot_data["display_sign"]):
        label.set_color(sign_colors[sign])
    ax.set_ylim(0, row_heights.sum())
    ax.set_xticks(np.arange(1, 10))
    ax.tick_params(axis="x", labelsize=12.5)
    ax.set_xlabel(r"$-\log_{10}$(adjusted P)", fontsize=11.5)
    fig.text(
        0.5,
        0.98,
        "Pathway enrichment of stable genes",
        ha="center",
        va="top",
        **PANEL_TITLE_STYLE,
    )
    fig.text(
        0.5,
        0.925,
        "All terms meet adjusted P < 0.05; point area indicates overlapping gene count",
        ha="center",
        va="top",
        fontsize=11,
        color="#5E6870",
    )

    # Enable grid lines horizontally and vertically to match screenshot style
    ax.grid(True, which='both', color='#D8DDE1', linestyle='-', linewidth=0.55, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    # Make all spines (borders) visible and black around the plot box
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)

    # Boundary line separating the two gene sets.
    for index, (_, end) in enumerate(boundaries[:-1]):
        ax.axhline(float(row_bottoms[end]), color="#A6A6A6", linewidth=0.8, zorder=2)

    # Compact legends inside the sparsest lower-right region.
    category_handles = [
        ax.scatter([], [], s=78, marker="o", color=POS_COLOR, edgecolor="white", label="Positive"),
        ax.scatter([], [], s=78, marker="o", color=NEG_COLOR, edgecolor="white", label="Negative"),
    ]
    category_legend = ax.legend(
        handles=category_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.985),
        ncol=2,
        frameon=False,
        fontsize=12.5,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0,
    )
    ax.add_artist(category_legend)

    legend_counts = [10, 40, 70]
    count_handles = [
        ax.scatter(
            [],
            [],
            s=34 + 4.6 * count,
            color="#8A8F94",
            edgecolor="white",
            linewidth=0.75,
            label=str(count),
        )
        for count in legend_counts
    ]
    ax.legend(
        handles=count_handles,
        title="Gene count",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.835),
        ncol=3,
        frameon=False,
        fontsize=11.5,
        title_fontsize=12,
        labelspacing=1.15,
        columnspacing=0.75,
        handletextpad=0.3,
        borderaxespad=0,
    )

    # Save the PDF deliverable.
    fig.subplots_adjust(left=0.47, right=0.985, top=0.875, bottom=0.10)
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    print(f"Saved PDF to {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
