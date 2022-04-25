
library(tidyverse)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../HelperFunctions/Defaults.R')

f <- function(x) (x > 0) * x
x <- seq(-5, 5, by = 0.1)
y <- f(x)

ggplot(mapping = aes(x, y)) +
  geom_line(size = 1) +
  theme_minimal() +
  labs(y = 'f(x)')
ggsave('relu.png',
       width = pic_width * 2/3,
       height = pic_height,
       units = pic_units,
       bg = 'white')
