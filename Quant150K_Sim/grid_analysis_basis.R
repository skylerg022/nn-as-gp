## Skyler Gray
## Show plots of validation loss performance across 
## basis grid search

# Load libraries
library(tidyverse)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

model <- read_csv('data/gridsearch_basis/lee2018/grid_basis_4by4.csv')
loss <- read_csv('data/gridsearch_basis/lee2018/grid_basis_4by4_val_mse.csv') %>%
  mutate(rmse = sqrt(val_mse)) %>%
  inner_join(model, by = 'model_num')

# Grid Search: General Trends ---------------------------------------------

# Filter down to RMSE less than 5
loss %>%
  group_by(model_num) %>%
  mutate(keep = sum(rmse < 1.5)) %>%
  ungroup() %>%
  filter(keep > 0) %>%
  ggplot(aes(x = epoch, y = rmse, group = model_num)) +
  # geom_smooth(se = FALSE, alpha = 0.5) +
  geom_line() +
  scale_y_continuous(limits = c(NA, 5))
# It seems that more than 30 epochs is necessary for better
#  performance

loss %>%
  ggplot(aes(x = epoch, y = rmse, group = model_num)) +
  geom_smooth(se = FALSE, alpha = 0.5, col = 'royalblue') +
  facet_grid(epochs ~ decay_rate) +
  scale_y_continuous(limits = c(NA, 5))
# Current decay rate settings may be too fast, don't seem to be helpful
# Consider reducing decay rate further from 


# 5 Best Models: Lowest RMSE Anywhere -------------------------------------

# Filter down to models with lowest loss at any epoch
min_loss_modelnum <- loss %>%
  group_by(model_num) %>%
  summarize(min_rmse = min(rmse)) %>%
  slice_min(min_rmse, n = 5) %>%
  pull(model_num)
loss_best <- loss %>%
  filter(model_num %in% min_loss_modelnum)
  
loss_best %>%
  ggplot(aes(x = epoch, y = rmse, 
             group = model_num, 
             col = as.factor(model_num))) +
  geom_line(alpha = 0.5) +
  geom_smooth(se = FALSE) +
  labs(col = 'Model', x = 'Epoch', y = 'RMSE') +
  theme_bw()

loss_best %>%
  group_by(model_num, n_layers, layer_width,
           batch_size, learning_rate, weight_decay,
           sigma_w, sigma_b) %>%
  summarize(min_rmse = round(min(rmse), 2),
            min_rmse_epoch = which.min(rmse)) %>%
  View()
