#!/usr/bin/env python3
"""Prepare Metascape multiple-list upload file for FDC stable genes."""

import csv
from itertools import zip_longest
from pathlib import Path
import sys


CODE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(CODE_DIR))
from cca401.preferred import CURRENT, selected_fit
FIT_DIR, _ = selected_fit('nc+fc')
BASE = CURRENT / "molecular" / FIT_DIR.relative_to(CURRENT / "fits")
OUT_DIR = CURRENT / "metascape" / "inputs"
TXT_OUT = OUT_DIR / "fdc_stable_gene_lists_metascape.txt"
CSV_OUT = OUT_DIR / "fdc_stable_gene_lists_metascape.csv"

LISTS = [
    ("CD1_FDC_positive", BASE / "combined_cd1_fdc" / "positive_genes.txt"),
    ("CD1_FDC_negative", BASE / "combined_cd1_fdc" / "negative_genes.txt"),
    ("CD2_FDC_positive", BASE / "combined_cd2_fdc" / "positive_genes.txt"),
    ("CD2_FDC_negative", BASE / "combined_cd2_fdc" / "negative_genes.txt"),
]


def write_multiple_list_csv(
    gene_lists: list[tuple[str, list[str]]], output: Path
) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([name for name, _ in gene_lists])
        writer.writerows(
            zip_longest(*(genes for _, genes in gene_lists), fillvalue="")
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gene_lists = []
    lines = ["#Name\tGenes"]
    for name, path in LISTS:
        genes = [gene.strip() for gene in path.read_text().splitlines() if gene.strip()]
        gene_lists.append((name, genes))
        lines.append(f"{name}\t{','.join(genes)}")
    TXT_OUT.write_text("\n".join(lines) + "\n")
    write_multiple_list_csv(gene_lists, CSV_OUT)
    print(TXT_OUT)
    print(CSV_OUT)
    print(f"gene_lists={len(LISTS)}")
    print(f"txt_bytes={TXT_OUT.stat().st_size}")
    print(f"csv_bytes={CSV_OUT.stat().st_size}")


if __name__ == "__main__":
    main()
