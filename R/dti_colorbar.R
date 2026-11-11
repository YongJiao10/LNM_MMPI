# ---- Libraries ----
library(ggplot2)
library(cowplot)
library(grid)       # unit()
library(scales)     # rescale()

# ---- Input ----
ranges_path <- "/Users/jy/Documents/cvae_mac/analysis_results/dti_colorbar_ranges.csv"

# 读取 vmin/vmax（文件需包含列：Subtype, vmin, vmax）
subtypes <- c("Externalizing-Disinhibited", "Internalizing-Distressed", "Mixed Dysregulation")
ranges <- read.csv(ranges_path, stringsAsFactors = FALSE)
ranges <- subset(ranges, Subtype %in% subtypes)
ranges$Subtype <- factor(ranges$Subtype, levels = subtypes)
ranges <- ranges[order(ranges$Subtype), ]

# ---- Helpers ----
make_fill <- function(vmin, vmax) {
  # 竖向色带；长度/宽度可调 barheight/barwidth
  scale_fill_gradientn(
    colours = c("mediumorchid3", "white", "goldenrod"),
    limits  = c(vmin, vmax),
    values  = rescale(c(vmin, 0, vmax)),
    name    = expression(paste("Cohen's ", italic(d))),  # 标题
    guide   = guide_colorbar(
      direction      = "vertical",
      barheight      = unit(2.4, "cm"),  # 缩短长度
      barwidth       = unit(0.6, "cm"),  # 细一些
      title.position = "top",
      title.hjust    = 0.5,
      ticks          = TRUE
    )
  )
}

legend_grob <- function(vmin, vmax) {
  df <- data.frame(x = c(vmin, vmax), y = 1)
  p <- ggplot(df, aes(x = x, y = y, fill = x)) +
    geom_tile() +
    make_fill(vmin, vmax) +
    theme_void() +
    theme(
      legend.position = "right",
      legend.title    = element_text(size = 10),
      legend.text     = element_text(size = 9),
      legend.margin   = margin(0, 0, 0, 0),
      plot.margin     = margin(0, 0, 0, 0)
    )
  cowplot::get_legend(p)
}

# ---- Build three vertical colorbars (no subtype titles) ----
grobs <- lapply(seq_len(nrow(ranges)), function(i) {
  legend_grob(ranges$vmin[i], ranges$vmax[i])
})

# 并排展示三个竖直色条（若想上下堆叠，改 nrow=3, ncol=1）
panel <- cowplot::plot_grid(plotlist = grobs, nrow = 1, rel_widths = rep(1, length(grobs)))
print(panel)
