library(dplyr)     
library(stringr)
library(ggplot2)
library(ggseg)
library(patchwork)
library(cowplot)  # 用于提取图例和精细布局

setwd("/Users/jy/Documents/cvae_mac/analysis_results")

process_data <- function(data) {
  data %>%
    mutate(
      # 提取半球信息
      hemi = case_when(
        grepl("^lh_", region) ~ "left",
        grepl("^rh_", region) ~ "right",
        grepl("^midline_", region) ~ "midline",  # 新增
        TRUE ~ NA_character_  # 无法识别的标记为NA
      ),
      
      # 清理区域名称（保留中线结构原名）
      region = ifelse(hemi %in% c("left", "right"),
                      gsub("^(lh_|rh_)", "", region),
                      gsub("^midline_", "", region))
    ) %>%
    filter(!is.na(hemi))  # 移除非标准名称
}

selected_subtype <- "Mixed Dysregulation"   # 'Externalizing-Disinhibited', 'Internalizing-Distressed', 'Mixed Dysregulation'

# 加载数据
vol <- read.csv("vol.csv") %>%
  filter(Subtype == selected_subtype) %>%
  process_data()

ct <- read.csv("ct.csv") %>%
  filter(Subtype == selected_subtype) %>%
  process_data()

csa <- read.csv("csa.csv") %>%
  filter(Subtype == selected_subtype) %>%
  process_data()

sub <- read.csv("subcor.csv") %>%
  filter(Subtype == selected_subtype) %>%
  process_data()

# 皮下若没有值添加一个空值
sub <- if (nrow(sub) > 0) {
  sub
} else {
  data.frame(
    region = 'hippocampus',  # 选择aseg中存在的皮下区域
    Cohens.d = 0, 
    hemi = 'left',
    stringsAsFactors = FALSE
  )
}

# 按指定顺序组合数据
group_order <- c(
  "Cortical Volume",
  "Cortical Thickness",
  "Cortical Surface Area"
)

# 若某个面板没值添加一个空值
combined_data <- bind_rows(
  if (nrow(vol) > 0) cbind(vol, group = "Cortical Volume") else 
    data.frame(region='cuneus', Cohens.d=0, hemi='left', group="Cortical Volume"),
  
  if (nrow(ct) > 0) cbind(ct, group = "Cortical Thickness") else 
    data.frame(region='cuneus', Cohens.d=0, hemi='left', group="Cortical Thickness"),
  
  if (nrow(csa) > 0) cbind(csa, group = "Cortical Surface Area") else 
    data.frame(region='cuneus', Cohens.d=0, hemi='left', group="Cortical Surface Area")
) %>%
  mutate(group = factor(group, levels = group_order))%>%
  group_by(group)


# 计算所有数据的共同范围
all_d_values <- c(combined_data$Cohens.d, sub$Cohens.d)  # 假设列名是Cohens.d
d_range <- range(all_d_values, na.rm = TRUE)

# 智能创建colorbar
common_fill <- if (min(d_range) >= 0) {
  # 情况1：数据全 ≥ 0
  scale_fill_gradientn(
    colours = c("white", "goldenrod"),  # 白→金渐变
    na.value = "white",
    limits = d_range,
    name = expression(paste("Cohen's ", italic(d)))
  )
} else if (max(d_range) <= 0) {
  # 情况2：数据全 ≤ 0
  scale_fill_gradientn(
    colours = c("mediumorchid3", "white"),  # 紫→白渐变
    na.value = "white",
    limits = d_range,
    name = expression(paste("Cohen's ", italic(d)))
  )
} else {
  # 情况3：数据有正有负
  scale_fill_gradientn(
    colours = c("mediumorchid3", "white", "goldenrod"),  # 紫→白→金
    na.value = "white",
    limits = d_range,
    name = expression(paste("Cohen's ", italic(d))),
    values = scales::rescale(c(min(d_range), 0, max(d_range)))
  )
}

# 皮层图（移除图例）
p_cortical <- ggseg(
  .data = combined_data,  
  colour = "black",
  mapping = aes(fill = Cohens.d)
) +
  theme_void() +
  common_fill +
  facet_wrap(~group, ncol = 1) +
  labs(title = "") +
  theme(
    strip.text = element_text(
      size = 16,
      color = "black",
      margin = margin(b = 10, t = 10)
    ),
    strip.placement = "outside",
    legend.position = "none",  # 移除图例
  )

# 皮层下图（移除图例）
p_sub <- ggseg(
  .data = sub,
  atlas = aseg,
  mapping = aes(fill = Cohens.d),
  colour = "black"
) +
  common_fill +
  theme_void() +
  labs(title = "Subcortical Volume") +
  theme(
    plot.title = element_text(size = 16, hjust = 0.5),
    legend.position = "none"  # 移除图例
  )

# 提取一个独立的图例
legend_plot <- ggplot(data.frame(x = d_range, y = 1), aes(x = x, y = y, fill = x)) +
  geom_tile() +
  common_fill +
  theme_void() +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
  )

# 获取图例
legend <- cowplot::get_legend(legend_plot)


# 拼合主图（调整 subcortical 面板高度）
main_plot <- p_cortical / p_sub +
  plot_layout(heights = c(3, 1))  # 皮层占3份高度，皮层下占1份

# 最终组合（主图 + 右侧图例）
p_final <- main_plot + 
  inset_element(legend,
                left = 0.85, right = 0.95,  # 图例水平位置（靠近右侧）
                bottom = 1.8, top=2.8)    # 图例垂直居中

print(p_final)