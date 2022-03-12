# Provide visualization for basis function expansions

library(tidyverse)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../HelperFunctions/defaults.R')
source('../HelperFunctions/Preprocess.R')
theme_set(theme_minimal())

load('../Quant150K_Sim/data/SimulatedTempsSplit.RData')

x_bases <- multiResBases(x_train = rbind(x_train, x_val), x_withheld = x_test,
                         sqrt_n_knots = c(4), thresh_type = 'colsum',
                         thresh = 30, thresh_max = 0.75)

train <- cbind(rbind(x_train, x_val), x_bases$x_train) %>%
  as.data.frame()

# Observe one of the bases
train %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(basis = mean(X4_6)) %>%
  ggplot(aes(x, y, fill = basis)) +
  geom_raster() +
  labs(fill = 'Basis6')

ggsave('basis_example.png',
       width = pic_width * 2/3,
       height = pic_height,
       units = pic_units,
       bg = 'white')
