library(ggseg)
library(ggsegSchaefer)
library(ggplot2)



p <- ggseg(
  atlas = schaefer7_100,
  mapping = aes(fill = region),
  position = "stacked",
  colour = "gray40",
  size   = 1
) +
  theme_void() +
  labs(fill = "")

print(p)
