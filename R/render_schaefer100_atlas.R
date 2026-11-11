#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(ggseg)
  library(ggsegSchaefer)
  library(scales)
})

# Read lines of label files
# Path relative to LNM_MMPI_code/
official_lines <- readLines("atlas/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_label.txt")
friendly_lines <- readLines("atlas/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_label_friendly.txt")

# Extract the 100 cortical labels (indices 17 to 116)
idx_cortical <- 17:116
official_names <- sapply(idx_cortical, function(i) official_lines[2 * i - 1])
friendly_names <- sapply(idx_cortical, function(i) friendly_lines[2 * i - 1])

# Clean names (remove whitespace)
official_names <- trimws(official_names)
friendly_names <- trimws(friendly_names)

# Create mapping dataframe
mapping_df <- data.frame(
  region = official_names,
  friendly_name = friendly_names,
  stringsAsFactors = FALSE
)

# Extract network from friendly name, e.g., Fusiform(VN/L) -> VN
mapping_df$network <- gsub(".*\\((.*)/.*\\).*", "\\1", mapping_df$friendly_name)

# Define network order (Yeo standard)
network_order <- c("VN", "SMN", "DAN", "VAN", "LN", "FPCN", "DMN")
mapping_df$network <- factor(mapping_df$network, levels = network_order)

# Sort mapping_df by network and friendly_name
mapping_df <- mapping_df[order(mapping_df$network, mapping_df$friendly_name), ]

# Base colors for the 7 networks
net_base_colors <- c(
  VN = "#783481",       # Purple
  SMN = "#4172b8",      # Blue
  DAN = "#2c7e11",      # Green
  VAN = "#dc5cff",      # Violet/Magenta
  LN = "#8a4718",       # Brown/Limbic
  FPCN = "#f7ac4a",     # Orange
  DMN = "#d85c6b"       # Red/Pink
)

# Function to generate shades for each network
get_shades <- function(base_color, n) {
  # Interpolate between a deeper shade (30% black) and a lighter shade (30% white)
  dark <- colorRampPalette(c("black", base_color))(10)[7]
  light <- colorRampPalette(c(base_color, "white"))(10)[4]
  colorRampPalette(c(dark, light))(n)
}

# Generate shaded colors for each of the 7 networks matching their region count
color_list <- list()
for (net in network_order) {
  n_regions <- sum(mapping_df$network == net)
  color_list[[net]] <- get_shades(net_base_colors[net], n_regions)
}

# Aligned Legend Padding Logic:
# Pad each network's levels to the maximum region count (24 for DMN)
# to achieve a perfect 7-column layout with no network spillover.
max_regions <- 24
all_levels <- c()
padded_color_palette <- c()
dummy_count <- 0

for (net in network_order) {
  net_df <- mapping_df[mapping_df$network == net, ]
  regions <- as.character(net_df$friendly_name)
  
  # Add real regions
  all_levels <- c(all_levels, regions)
  
  net_colors <- color_list[[net]]
  names(net_colors) <- regions
  padded_color_palette <- c(padded_color_palette, net_colors)
  
  # Add dummy regions to pad to max_regions
  n_dummies <- max_regions - length(regions)
  if (n_dummies > 0) {
    for (i in 1:n_dummies) {
      dummy_count <- dummy_count + 1
      dummy_name <- paste(rep(" ", dummy_count), collapse = "")
      all_levels <- c(all_levels, dummy_name)
      
      dummy_color <- "transparent"
      names(dummy_color) <- dummy_name
      padded_color_palette <- c(padded_color_palette, dummy_color)
    }
  }
}

# Apply all levels (including dummies) to the mapping_df factor
mapping_df$friendly_name <- factor(mapping_df$friendly_name, levels = all_levels)

# Build cortical plot
p <- ggseg(
  .data = mapping_df,
  atlas = schaefer7_100,
  mapping = aes(fill = friendly_name),
  position = "stacked",
  colour = "gray30",
  linewidth   = 0.3
) +
  scale_fill_manual(values = padded_color_palette, limits = all_levels, drop = FALSE) +
  theme_void() +
  # Add text labels above the hemispheres to mark Left/Right and Lateral/Medial views
  # View coordinates under position="stacked":
  # Left Lateral: (0-293, 0-204); Left Medial: (350-655, 0-197)
  # Right Lateral: (0-291, 260-466); Right Medial: (350-655, 260-462)
  # Adjust text sizes as requested (increased from 8 to 16)
  annotate("text", x = c(146.5, 502.5, 146.5, 502.5), y = c(225, 225, 500, 500),
           label = c("Left Lateral", "Left Medial", "Right Lateral", "Right Medial"),
           size = 16, fontface = "bold", hjust = 0.5, colour = "black") +
  ylim(-20, 540) +
  theme(
    legend.position = "bottom",
    # Increase legend font sizes by 4 points as requested (13 -> 17, 16 -> 20)
    legend.text = element_text(size = 17, face = "bold"),
    legend.title = element_text(size = 20, face = "bold"),
    legend.key.size = unit(0.6, "cm"),
    legend.spacing.x = unit(0.3, "cm"),
    legend.spacing.y = unit(0.15, "cm"),
    legend.key = element_blank(),
    plot.margin = margin(20, 20, 20, 20)
  ) +
  # Guides with 7 columns (one for each network), filled column-by-column
  guides(fill = guide_legend(ncol = 7, byrow = FALSE, title.position = "top", title.hjust = 0.5)) +
  labs(fill = "Cortical Regions (Schaefer 100) - Grouped by Networks (VN, SMN, DAN, VAN, LN, FPCN, DMN) in Shades")

# Create figs output dir if not exist
dir.create("figs", showWarnings = FALSE, recursive = TRUE)

# Save high-resolution outputs with larger canvas to accommodate larger fonts
ggsave("figs/schaefer100_cortical_atlas.png", plot = p, width = 28, height = 30, dpi = 300)
ggsave("figs/schaefer100_cortical_atlas.pdf", plot = p, width = 28, height = 30, device = "pdf")

message("Successfully saved cortical atlas to figs/schaefer100_cortical_atlas.png and figs/schaefer100_cortical_atlas.pdf")
