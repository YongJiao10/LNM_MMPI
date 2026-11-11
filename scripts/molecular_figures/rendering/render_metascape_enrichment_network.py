#!/usr/bin/env python3
"""Export the cropped Metascape network and its separate cluster legend."""

import argparse
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

CODE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIR))
from cca401 import preferred

CLUSTER_COLORS = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#FFFF33", "#A65628", "#F781BF", "#999999", "#8DD3C7",
    "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462",
    "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path,
        default=preferred.CURRENT / "metascape" / "cd1_only_20261102",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=CODE_DIR / "figs" / "supp_gene_stability",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.source_dir / "_FINAL_GO.csv")
    representatives = table.loc[
        table["FirstInGroupByLogP"].eq(1) & table["GROUP_ID"].between(1, 20)
    ].sort_values("GROUP_ID")
    assert representatives["GROUP_ID"].tolist() == list(range(1, 21))
    labels = [
        textwrap.wrap(term, width=48, break_long_words=False, break_on_hyphens=False)
        for term in representatives["Description"]
    ]

    # The website legend is a clipped image. Rebuild only its labels and keys,
    # using the full saved descriptions and the original twenty cluster colors.
    width = 592.0
    line_height, row_gap, margin = 11.5, 3.5, 12.0
    column_heights = [
        sum(len(lines) * line_height + row_gap for lines in labels[start:start + 10])
        for start in (0, 10)
    ]
    legend_height = max(column_heights) + 2 * margin
    plt.rcParams.update({"pdf.fonttype": 42})
    fig = plt.figure(figsize=(width / 72, legend_height / 72))
    for column, start in enumerate((0, 10)):
        x = margin + column * width / 2
        y = legend_height - margin
        for index in range(start, start + 10):
            fig.patches.append(Rectangle(
                (x / width, (y - 8) / legend_height), 7 / width, 7 / legend_height,
                transform=fig.transFigure, facecolor=CLUSTER_COLORS[index],
                edgecolor="none",
            ))
            fig.text(
                (x + 12) / width, y / legend_height,
                "\n".join(labels[index]), va="top", ha="left",
                fontsize=10, linespacing=1.15,
            )
            y -= len(labels[index]) * line_height + row_gap
    legend_output = args.output_dir / "fdc_metascape_enrichment_legend.pdf"
    fig.savefig(legend_output, facecolor="white")
    plt.close(fig)

    # Keep the website network's vector content and layout intact. Crop only;
    # the author will place the separate legend in the lower right in slides.
    source = PdfReader(args.source_dir / "ColorByCluster.pdf")
    network = source.pages[0]
    bounds = (10, 585, 602, 798)
    network.cropbox = RectangleObject(bounds)
    network.mediabox = RectangleObject(bounds)
    writer = PdfWriter()
    writer.add_page(network)
    output = args.output_dir / "fdc_metascape_enrichment_network.pdf"
    writer.write(output)
    print(f"Saved: {output}")
    print(f"Saved: {legend_output}")


if __name__ == "__main__":
    main()
