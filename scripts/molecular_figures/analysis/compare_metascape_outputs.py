"""Compare authorized old/new Metascape outputs without rerunning enrichment.

Read the selected, authorized standalone or extracted result files.
The 3000-gene threshold is the documented service rule, not a new analysis.
"""

import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import pandas as pd


def read_run(folder, html_name):
    evidence = pd.read_csv(folder / "Evidence/Evidence.csv")
    terms = pd.read_csv(folder / "Enrichment_GO/GO_AllLists.csv")
    combined = pd.read_csv(folder / "Enrichment_GO/_FINAL_GO.csv")
    hit_cols = [c for c in evidence if c.startswith("EvidenceHit:")]
    evidence_cols = [c for c in evidence if c.startswith("Evidence")]
    counts = evidence.TotalEvidenceCounts.value_counts().sort_index(ascending=False)
    cumulative = counts.cumsum()
    threshold = float(cumulative[cumulative <= 3000].index[-1])
    selected = set(evidence.loc[evidence.TotalEvidenceCounts >= threshold, "Gene"].astype(str))
    memberships = {
        c.split(":", 1)[1]: set(evidence.loc[evidence[c] == 1, "Gene"].astype(str))
        for c in hit_cols
    }
    individual_sizes = set(terms["#GeneInHitList"].unique())
    pooled_sizes = set(combined["#GeneInHitList"].unique()) - individual_sizes
    assert len(pooled_sizes) == 1
    pooled_size = int(pooled_sizes.pop())
    pooled = combined.loc[combined["#GeneInHitList"] == pooled_size]
    ns = {"g": "http://www.cs.rpi.edu/XGMML"}
    graph = ET.parse(folder / "Enrichment_GO/GONetwork.xgmml").getroot()
    nodes = [
        {**n.attrib, **{x.get("name"): x.get("value") for x in n.findall("g:att", ns)}}
        for n in graph.findall("g:node", ns)
    ]
    edges = [
        {**n.attrib, **{x.get("name"): x.get("value") for x in n.findall("g:att", ns)}}
        for n in graph.findall("g:edge", ns)
    ]
    html = (folder / html_name).read_text()
    soup = BeautifulSoup(html, "html.parser")
    table = [
        t for t in soup.find_all("table")
        if [x.get_text(strip=True) for x in t.find_all("th")][:3] == ["Name", "Total", "Unique"]
    ][0]
    input_stats = {
        cells[0]: {"raw": int(cells[1]), "unique_entrez": int(cells[2])}
        for row in table.find_all("tr")[1:]
        for cells in [[x.get_text(strip=True) for x in row.find_all("td")]]
    }
    top20 = combined.loc[combined.FirstInGroupByLogP == 1].nsmallest(20, "LogP")
    summary = {
        "database_date": re.search(r"last updated on (\d{4}-\d{2}-\d{2})", html).group(1),
        "input_stats": input_stats,
        "input_union_entrez": len(evidence),
        "evidence_columns": evidence_cols,
        "ppi_evidence_columns": [c for c in evidence_cols if c.startswith("EvidencePPI:")],
        "evidence_sum_max_error": float((evidence[evidence_cols].sum(axis=1) - evidence.TotalEvidenceCounts).abs().max()),
        "evidence_counts": {str(k): int(v) for k, v in counts.items()},
        "selected_threshold_under_3000": threshold,
        "selected_count": len(selected),
        "pooled_annotated_input_count": pooled_size,
        "pooled_rows": len(pooled),
        "pooled_rows_with_genes_outside_selected": sum(
            not set(s.split("|")).issubset(selected) for s in pooled.GeneID
        ),
        "individual_rows": len(terms),
        "individual_counts": terms.groupby("GeneList").size().to_dict(),
        "combined_rows": len(combined),
        "combined_clusters": int(combined.GROUP_ID.nunique()),
        "network_nodes": len(nodes), "network_edges": len(edges),
        "edge_similarity_min": min(float(x["SCORE"]) for x in edges),
        "network_terms_not_in_table": sorted({n["label"] for n in nodes} - set(combined.GO)),
        "top20": top20[["GO", "Description", "GROUP_ID", "LogP", "Log(q-value)"]].to_dict("records"),
        "individual_top10": {
            name: rows.nsmallest(10, "LogP")[["GO", "Description", "LogP", "Log(q-value)"]].to_dict("records")
            for name, rows in terms.groupby("GeneList")
        },
    }
    return summary, selected, memberships, combined, {n["label"] for n in nodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_dir", type=Path)
    parser.add_argument("new_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--old-session", required=True)
    parser.add_argument("--new-session", required=True)
    parser.add_argument("--old-html", required=True)
    parser.add_argument("--new-html", required=True)
    parser.add_argument("--comparison-context", required=True)
    args = parser.parse_args()
    old, old_sel, old_groups, old_terms, old_nodes = read_run(args.old_dir, args.old_html)
    new, new_sel, new_groups, new_terms, new_nodes = read_run(args.new_dir, args.new_html)
    old20 = {x["GO"] for x in old["top20"]}
    new20 = {x["GO"] for x in new["top20"]}
    comparison = {
        "old_session": args.old_session, "new_session": args.new_session, "old": old, "new": new,
        "selected_genes_common": len(old_sel & new_sel),
        "selected_genes_old_only": len(old_sel - new_sel),
        "selected_genes_new_only": len(new_sel - old_sel),
        "individual_gene_membership": {
            name: {"old": len(a), "new": len(new_groups[name]), "common": len(a & new_groups[name]),
                   "old_only": len(a - new_groups[name]), "new_only": len(new_groups[name] - a)}
            for name, a in old_groups.items()
        },
        "top20_shared_ids": sorted(old20 & new20),
        "top20_old_only_ids": sorted(old20 - new20),
        "top20_new_only_ids": sorted(new20 - old20),
        "network_shared_terms": len(old_nodes & new_nodes),
        "network_old_only_terms": len(old_nodes - new_nodes),
        "network_new_only_terms": len(new_nodes - old_nodes),
        "comparison_context": args.comparison_context,
        "limitation": "Descriptive comparison of saved outputs. No enrichment reruns or new inference; differences in exported P values are not tests of differences between runs.",
    }
    args.out_dir.mkdir(exist_ok=True, parents=True)
    (args.out_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    top = pd.concat([
        pd.DataFrame(old["top20"]).assign(version="old"),
        pd.DataFrame(new["top20"]).assign(version="new"),
    ])
    top.to_csv(args.out_dir / "top20_old_new.csv", index=False)
    common_terms = old_terms[["GO", "Description", "LogP", "#GeneInHitList"]].merge(
        new_terms[["GO", "Description", "LogP", "#GeneInHitList"]], on="GO", how="outer", suffixes=("_old", "_new")
    )
    common_terms.to_csv(args.out_dir / "combined_terms_old_new.csv", index=False)
    print(json.dumps({k: v for k, v in comparison.items() if k not in ("old", "new")}, ensure_ascii=False, indent=2))
    for label, result in (("old", old), ("new", new)):
        print(label, json.dumps({k: v for k, v in result.items() if k not in ("individual_top10", "top20")}, ensure_ascii=False, indent=2))
        print(label, "individual_top10", json.dumps(result["individual_top10"], ensure_ascii=False))


if __name__ == "__main__":
    main()
