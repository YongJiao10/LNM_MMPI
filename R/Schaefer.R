library(ggseg)
library(ggsegSchaefer)
library(ggplot2)
# --- Schaefer 7_100, 左外侧，按7个网络自定义颜色，无需 geom_sf ---
suppressPackageStartupMessages({
  library(ggseg)
  library(ggsegSchaefer)
  library(ggplot2)
})

# 1) 7个网络颜色
net_cols <- c(
  Vis="#773481", SomMot="#6396c2", DorsAttn="#2c7e11",
  SalVentAttn="#dc5cff", Limbic="#8a4718", Cont="#f7ac4a", Default="#d85c6b"
)

# 2) 取底层数据（兼容不同版本）
atlas <- schaefer7_100
df <- tryCatch(atlas$data, error=function(e) NULL)
if (is.null(df)) df <- as.data.frame(atlas)

# 3) 选择用于解析网络名的源列
if ("network" %in% names(df)) {
  src <- df$network
} else if ("annot" %in% names(df)) {
  src <- df$annot
} else if ("label" %in% names(df)) {
  src <- df$label
} else {
  src <- df$region
}

s <- tolower(as.character(src))

# 4) 解析7个网络（逐步赋值，避免多层 ifelse 括号）
net <- rep(NA_character_, length(s))
net[grepl("vis", s)]                                    <- "Vis"
net[grepl("som\\s*mot", s)]                             <- "SomMot"
net[grepl("dors\\s*attn", s)]                           <- "DorsAttn"
net[grepl("(sal|vent).*?(attn|salience)", s)]           <- "SalVentAttn"
net[grepl("limbic", s)]                                 <- "Limbic"
net[grepl("cont|control|fpn|frontoparietal", s)]        <- "Cont"
net[grepl("default|dmn", s)]                            <- "Default"

# 5) 构造 region -> 颜色（未匹配的置灰）
reg_levels <- unique(as.character(df$region))
first_non_na <- function(v){ v <- v[!is.na(v)]; if (length(v)) v[1] else NA_character_ }
reg2net <- tapply(net, as.character(df$region), first_non_na)

pal_region <- setNames(rep("grey85", length(reg_levels)), reg_levels)
hit <- intersect(names(pal_region), names(reg2net))
pal_region[hit] <- net_cols[ reg2net[hit] ]

# 6) 画左半球-外侧；size 控制边界粗细（无需额外 geom_sf）
p <- ggseg(
  atlas = schaefer7_100,
  hemisphere = "left",
  view = "lateral",
  mapping = aes(fill = region),
  colour = "gray40",
  size   = 1.2
) +
  scale_fill_manual(values = pal_region, na.value = "grey85") +
  theme_void() +
  theme(legend.position = "none")

# 7) 保存矢量 PDF 到 flowchart/subpanels
get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  match <- grep("--file=", args)
  if (length(match) > 0) return(sub("--file=", "", args[match]))
  match_f <- which(args == "-f")
  if (length(match_f) > 0 && length(args) >= match_f + 1) return(args[match_f + 1])
  return(getwd())
}
script_path <- get_script_path()
script_dir <- dirname(normalizePath(script_path))
code_root <- if (basename(script_dir) == "R") dirname(script_dir) else script_dir
project_root <- normalizePath(file.path(code_root, ".."))
flowchart_dir <- file.path(project_root, "flowchart", "subpanels")
dir.create(flowchart_dir, showWarnings = FALSE, recursive = TRUE)
out_pdf <- file.path(flowchart_dir, "cortical-atlas.pdf")
ggsave(out_pdf, plot = p, width = 5, height = 3.8, bg = "transparent", device = "pdf")

# 裁切多余白边
pdfcrop_bin <- "/Library/TeX/texbin/pdfcrop"
if (file.exists(pdfcrop_bin)) {
  system2(pdfcrop_bin, c(out_pdf, out_pdf), stdout = FALSE, stderr = FALSE)
}

print(p)

