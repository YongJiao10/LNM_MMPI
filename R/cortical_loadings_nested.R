#!/usr/bin/env Rscript

# Render the adopted Figure 2–4 cortical maps and Figure 4 shared colorbars.
# Called by plots.py::plot_brain_maps from the corresponding main notebook.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(ggseg)
  library(ggsegSchaefer)
  library(cowplot)
  library(scales)
  library(grid)
})

parse_cli <- function(args) {
  if (length(args) != 6 || args[[1]] != "--input" ||
      args[[3]] != "--output-dir" || args[[5]] != "--model") {
    stop(
      "Usage: cortical_loadings_nested.R --input <cortical-loadings.csv> --output-dir <figs-dir> --model <nc|fc|nc+fc>",
      call. = FALSE
    )
  }
  list(input = args[[2]], output_dir = args[[4]], model = args[[6]])
}

make_fill_scale <- function(vmin, vmax, breaks = waiver(), labels = waiver()) {
  if (vmin >= 0) {
    scale_fill_gradientn(
      colours = c("white", "#c8971c"),
      limits = c(vmin, vmax),
      breaks = breaks,
      labels = labels,
      na.value = "grey95",
      name = ""
    )
  } else if (vmax <= 0) {
    scale_fill_gradientn(
      colours = c("#835bd6", "white"),
      limits = c(vmin, vmax),
      breaks = breaks,
      labels = labels,
      na.value = "grey95",
      name = ""
    )
  } else {
    scale_fill_gradientn(
      colours = c("#835bd6", "white", "#c8971c"),
      limits = c(vmin, vmax),
      values = scales::rescale(c(vmin, 0, vmax)),
      breaks = breaks,
      labels = labels,
      na.value = "grey95",
      name = ""
    )
  }
}

build_colorbar_only_plot <- function(vmin, vmax) {
  bar_breaks <- if (vmin < 0 && vmax > 0) c(vmin, 0, vmax) else c(vmin, vmax)
  fill_scale <- make_fill_scale(
    vmin,
    vmax,
    breaks = bar_breaks,
    labels = sprintf("%.1f", bar_breaks)
  )

  legend_seed <- ggplot(
    data.frame(x = seq(vmin, vmax, length.out = 256), y = 1),
    aes(x = x, y = y, fill = x)
  ) +
    geom_raster() +
    fill_scale +
    guides(
      fill = guide_colorbar(
        direction = "horizontal",
        barwidth = unit(28, "cm"),
        barheight = unit(1.6, "cm"),
        title = NULL,
        ticks = TRUE
      )
    ) +
    theme_void() +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.key.width = unit(8, "cm"),
      legend.key.height = unit(1.6, "cm"),
      legend.text = element_text(size = 48, colour = "black"),
      legend.title = element_blank(),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0)
    )

  cowplot::ggdraw(cowplot::get_legend(legend_seed))
}

options <- parse_cli(commandArgs(trailingOnly = TRUE))
if (!file.exists(options$input)) {
  stop(sprintf("Missing input CSV: %s", options$input), call. = FALSE)
}

dir.create(options$output_dir, showWarnings = FALSE, recursive = TRUE)
input_path <- normalizePath(options$input)
output_dir <- normalizePath(options$output_dir)
loadings <- read.csv(input_path, stringsAsFactors = FALSE)
figure_number <- switch(options$model, "nc" = 2L, "fc" = 3L, "nc+fc" = 4L)
if (is.null(figure_number)) {
  stop("Model must be nc, fc or nc+fc", call. = FALSE)
}
if (figure_number != 4L) {
  loadings <- loadings[loadings$figure == figure_number, ]
  loadings$cd <- loadings$dimension
  loadings$modality <- if (figure_number == 2L) "struct" else "func"
}

required_columns <- c("cd", "modality", "region_index", "value", "vmin", "vmax")
if (!all(required_columns %in% colnames(loadings))) {
  stop(
    sprintf(
      "Input CSV must contain columns: %s",
      paste(required_columns, collapse = ", ")
    ),
    call. = FALSE
  )
}

loadings <- loadings %>%
  mutate(
    cd = as.integer(cd),
    modality = as.character(modality),
    region_index = as.integer(region_index),
    value = as.numeric(value),
    vmin = as.numeric(vmin),
    vmax = as.numeric(vmax)
  )

if (!all(loadings$cd %in% c(1L, 2L))) {
  stop("Input CSV contains an invalid canonical-dimension index", call. = FALSE)
}
if (!all(loadings$modality %in% c("struct", "func"))) {
  stop("Input CSV contains an invalid modality; expected struct or func", call. = FALSE)
}
if (any(!is.finite(unlist(loadings[required_columns[c(4, 5, 6)]])))) {
  stop("Input CSV contains non-finite loading or color-range values", call. = FALSE)
}

expected_outputs <- character()
labels <- as.data.frame(schaefer7_100) %>%
  filter(grepl("7Networks", label)) %>%
  distinct(label) %>% arrange(label) %>% pull(label)
stopifnot(length(labels) == 100)
stopifnot(nrow(loadings) == if (figure_number == 4L) 400L else 200L)

network_colors <- c(
  Vis = "#5E5B8A", SomMot = "#4F8FB5", DorsAttn = "#83934E",
  SalVentAttn = "#A25C8A", Limbic = "#8a4718",
  Cont = "#C79A3B", Default = "#B24A43"
)
networks <- sub(".*7Networks_[LR]H_([^_]+)_.*", "\\1", labels)

for (cd_index in c(1L, 2L)) {
  cd_range <- loadings %>%
    filter(.data$cd == cd_index) %>%
    distinct(.data$vmin, .data$vmax)
  if (nrow(cd_range) != 1L || cd_range$vmin[[1]] >= cd_range$vmax[[1]]) {
    stop(sprintf("CD%d has an invalid shared color range", cd_index), call. = FALSE)
  }

  for (modality in unique(loadings$modality)) {
    rows <- loadings[loadings$cd == cd_index & loadings$modality == modality, ]
    rows <- rows[order(rows$region_index), ]
    stopifnot(nrow(rows) == 100, all(rows$region_index == 0:99))
    bounds <- unique(rows[c("vmin", "vmax")])
    stopifnot(nrow(bounds) == 1)
    panel <- ggseg(
      .data = data.frame(
        label = labels, value = rows$value,
        outline_color = network_colors[networks]
      ), atlas = schaefer7_100,
      mapping = aes(fill = value, colour = outline_color), size = 0.35
    ) + scale_colour_identity(na.value = "black") + scale_fill_gradientn(
      colours = c("#835bd6", "white", "#c8971c"),
      limits = c(bounds$vmin, bounds$vmax),
      values = scales::rescale(c(bounds$vmin, 0, bounds$vmax)),
      na.value = "white", name = ""
    ) + theme_void() + theme(legend.position = "none", plot.margin = margin(6, 6, 6, 6))

    if (figure_number == 4L) {
      model <- if (modality == "struct") "sLNM" else "fLNM"
      cortical_path <- file.path(output_dir, sprintf("brain116_cd%d_%s_cortical.pdf", cd_index, model))
      ggsave(cortical_path, plot = panel, width = 10,
             height = 18 * (1 / (4 + 3 * 0.08)), device = "pdf")
    } else {
      cortical_path <- file.path(output_dir, sprintf("fig%d_cd%d_cortical.pdf", figure_number, cd_index))
      ggsave(cortical_path, plot = panel, width = 10,
             height = 18 * (1 / (4 + 3 * 0.08)), device = "pdf", bg = "transparent")
    }
    expected_outputs <- c(expected_outputs, cortical_path)
  }

  if (figure_number == 4L) {
    colorbar_path <- file.path(output_dir, sprintf("colorbar_cd%d.pdf", cd_index))
    ggsave(
      colorbar_path,
      plot = build_colorbar_only_plot(cd_range$vmin[[1]], cd_range$vmax[[1]]),
      width = 12,
      height = 3,
      device = "pdf"
    )
    expected_outputs <- c(expected_outputs, colorbar_path)
  }
}

if (!all(file.exists(expected_outputs))) {
  stop("Cortical renderer did not create all expected PDFs", call. = FALSE)
}
message("Saved Figure ", figure_number, " cortical maps and shared colorbars to: ", output_dir)
