library(ggseg)
library(ggplot2)

# print(plot(dk))

p <- ggseg(atlas = dk,
           mapping = aes(fill = region),
           # colour = "black",  # 轮廓线颜色
           position = "stacked") +      # 轮廓线粗细
  scale_fill_brain("dk") +    # 使用dk图谱默认配色
  theme_void() +
  labs(fill = "")


print(plot(p))

ggsave("dk_atlas.pdf", 
       plot = p,
       width = 8,  # 宽度（英寸）
       height = 6, # 高度（英寸）
       device = "pdf")