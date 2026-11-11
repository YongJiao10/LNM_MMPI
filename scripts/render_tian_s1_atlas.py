#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from subcortex_visualization.plotting import plot_subcortical_data

# The 8 unique subcortical region names in the Tian S1 atlas
# SC stands for Subcortex / Subcortical, differentiating them from cortical regions.
region_names_friendly_8 = [
    "Hippocampus",
    "Amygdala",
    "Thalamus_Posterior",
    "Thalamus_Anterior",
    "Accumbens",
    "Pallidum",
    "Putamen",
    "Caudate"
]

# The corresponding 8 official names used in subcortex_visualization mapping
region_order = [
    'hippocampus',
    'amygdala',
    'thalamus_posterior',
    'thalamus_anterior',
    'accumbens',
    'pallidum',
    'putamen',
    'caudate'
]

# Prepare plot_df
data_list = []
cmap_name = 'tab10'
cmap = plt.get_cmap(cmap_name)

# Create 16 values (repeating 0-7 for Left and Right hemispheres)
# so the same region in both hemispheres receives the exact same color.
for i in range(16):
    hemi = 'R' if i < 8 else 'L'
    region_name = region_order[i % 8]
    data_list.append({
        'region': region_name,
        'value': i % 8,  # Map LHS and RHS of the same region to the same color index
        'Hemisphere': hemi
    })

plot_df = pd.DataFrame(data_list)

# Generate the figure without the default continuous legend
fig = plot_subcortical_data(
    subcortex_data=plot_df,
    atlas='Tian_S1',
    hemisphere='both',
    show_legend=False,
    show_figure=False,
    cmap=cmap_name,
    vmin=0,
    vmax=7
)

# Get the main axis from the generated figure
ax = fig.axes[0]

# Add labels above each view to mark Left/Right and Medial/Lateral views
# X-centers for the four views: Left Medial (33.5), Left Lateral (95.7), Right Lateral (156.5), Right Medial (218.7)
# With y-axis inverted, Y = -4 is above the top of the brain (Y = 2)
# View labels font size set to 22 (increased by 8 points)
ax.text(33.5, -4, "Left Medial", ha="center", va="bottom", fontsize=22, fontweight="bold", color="black")
ax.text(95.7, -4, "Left Lateral", ha="center", va="bottom", fontsize=22, fontweight="bold", color="black")
ax.text(156.5, -4, "Right Lateral", ha="center", va="bottom", fontsize=22, fontweight="bold", color="black")
ax.text(218.7, -4, "Right Medial", ha="center", va="bottom", fontsize=22, fontweight="bold", color="black")

# Add a custom discrete legend for the 8 unique regions matching the mapped colors
legend_elements = []
for i in range(8):
    # Normalize to [0, 1] to get the exact color from the qualitative map
    color = cmap(i / 7.0)
    legend_elements.append(
        Patch(facecolor=color, edgecolor='black', label=region_names_friendly_8[i])
    )

# Place legend at the bottom with 4 columns (2 rows total)
# Changed loc to 'upper center' and bbox_to_anchor to (0.5, -0.15) to guarantee no overlap with the brain shapes.
# Legend font size is 17 (increased by 4) and title font size is 20 (increased by 4).
ax.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=4,
    frameon=False,
    fontsize=17,
    handleheight=1.2,
    handlelength=1.2,
    title="Subcortical Regions (Tian S1)",
    title_fontsize=20,
    handletextpad=0.4
)

# Adjust spacing to accommodate the labels and legend (increased bottom margin to 0.55 and reduced top to 0.88)
fig.subplots_adjust(bottom=0.55, top=0.88)

# Create figs output directory if it doesn't exist
os.makedirs("figs", exist_ok=True)

# Save high-resolution outputs
fig.savefig("figs/tian_s1_subcortex_atlas.png", dpi=300, bbox_inches='tight')
fig.savefig("figs/tian_s1_subcortex_atlas.pdf", bbox_inches='tight')

print("Successfully saved subcortical atlas to figs/tian_s1_subcortex_atlas.png and figs/tian_s1_subcortex_atlas.pdf")
