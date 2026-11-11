#!/usr/bin/env python3
"""Render Figure 5b from the adopted combined-model fit."""

import argparse
from collections import OrderedDict
import os
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cca401 import preferred
from plots import PAIR_COLOR, PANEL_TITLE_STYLE

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT / "figs" / "fig5")
output_dir = parser.parse_args().output_dir

fit, _ = preferred.selected_fit("nc+fc")
with np.load(fit / "model.npz") as model:
    U_fc_ref_fix = model["brain"].copy()

NETWORKS = ['SC', 'VN', 'SMN', 'DAN', 'VAN', 'LN', 'FPCN', 'DMN']
atlas_path = ROOT / 'atlas' / 'Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_label_friendly.txt'
lines = [ln.strip() for ln in atlas_path.read_text().splitlines() if ln.strip()]
atlas_labels = lines[::2]
pattern = re.compile(r'\(([A-Za-z0-9]+)/(?:L|R)\)\s*$')
net_to_idx = {net: [] for net in NETWORKS}
for i, label in enumerate(atlas_labels):
    net = pattern.search(label).group(1).upper()
    net_to_idx[net].append(i)

contribution_data = []
for d in (0, 1):
    w = U_fc_ref_fix[:, d]
    w_s = np.abs(w[:116])
    w_f = np.abs(w[116:])
    s_mean = OrderedDict((net, float(np.mean(w_s[net_to_idx[net]]))) for net in NETWORKS)
    f_mean = OrderedDict((net, float(np.mean(w_f[net_to_idx[net]]))) for net in NETWORKS)
    den = float(sum(s_mean.values()) + sum(f_mean.values()))
    s_share = float(sum(s_mean.values()) / den * 100)
    f_share = float(sum(f_mean.values()) / den * 100)
    print(f'CD{d+1}: Structure={s_share:.6f}%, Function={f_share:.6f}%')
    contribution_data.append({'Dimension': f'CD{d+1}', 'Modality': 'Structure', 'Contribution': s_share})
    contribution_data.append({'Dimension': f'CD{d+1}', 'Modality': 'Function', 'Contribution': f_share})
df_contrib = pd.DataFrame(contribution_data)

# Original Figure 5b plotting cell.
save_path = str(output_dir / 'combined_contrib_bar')
with sns.plotting_context("talk"), sns.axes_style("whitegrid"):
    fig = plt.figure(figsize=(6, 5.6))

    colors = {'Structure': PAIR_COLOR[0][0], 'Function': PAIR_COLOR[0][1]}
    ax = sns.barplot(data=df_contrib, y='Dimension', x='Contribution', hue='Modality',
                     palette=colors, linewidth=1.5)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='center',
                     color='white', fontweight='bold', fontsize=13)

    fig.suptitle('Structural and functional\nloading contributions', x=0.5,
                 y=0.98, ha='center', **{**PANEL_TITLE_STYLE, 'fontsize': 21.38})
    plt.xlabel('Relative Contribution (%)', fontsize=14)
    plt.ylabel('')
    plt.xticks(fontsize=11)
    plt.xlim(0, 60)
    plt.legend(bbox_to_anchor=(0.5, -0.16), loc='upper center', ncol=2, frameon=False, fontsize=17)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.81, bottom=0.20)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.show()
