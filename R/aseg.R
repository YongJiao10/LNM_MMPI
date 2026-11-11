library(ggseg)
library(ggplot2)

# 绘制 aseg 图谱（皮层下结构）
p_aseg <- ggseg(
  atlas = aseg,                   # 使用aseg图谱
  mapping = aes(fill = region),   # 按脑区填色
  # colour = "black",               # 轮廓线颜色
  # size = 0.3                      # 轮廓线粗细
) +
  scale_fill_brain("aseg") +      # 使用aseg默认配色
  theme_void() +
  labs(fill = "")

print(p_aseg)

ggsave("aseg_atlas.pdf", 
      plot = p_aseg,
      width = 8,  # 宽度（英寸）
      height = 6, # 高度（英寸）
      device = "pdf")