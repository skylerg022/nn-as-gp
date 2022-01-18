## Skyler Gray
## Show plots of validation loss performance across 
## neural net grid search

# Load libraries
library(tidyverse)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

model <- read_csv('data/gridsearch_nn/grid_nn.csv')
loss <- read_csv('data/gridsearch_nn/grid_nn_val_mse.csv') %>%
  mutate(rmse = sqrt(val_mse)) %>%
  inner_join(model, by = 'model_num')

# Grid Search: General Trends ---------------------------------------------

# Filter down to RMSE less than 5
loss %>%
  ggplot(aes(x = epoch, y = rmse, group = model_num)) +
  geom_smooth(se = FALSE, alpha = 0.5) +
  facet_grid(epochs ~ layer_width) +
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
  geom_smooth(se = FALSE) +
  labs(col = 'Model', x = 'Epoch', y = 'RMSE') +
  theme_bw()

loss_best %>%
  group_by(model_num, n_layers, layer_width, epochs,
           batch_size, decay_rate, dropout_rate) %>%
  summarize(min_rmse = round(min(rmse), 2),
            min_rmse_epoch = which.min(rmse)) %>%
  View()


# 5 Best Models: Lowest Ending RMSE -------------------------------------

best_end_modelnum <- loss %>%
  group_by(model_num) %>%
  filter(epoch == max(epoch)) %>%
  ungroup() %>%
  slice_min(rmse, n = 5) %>%
  pull(model_num)

loss %>%
  filter(model_num %in% best_end_modelnum) %>%
  ggplot(aes(x = epoch, y = rmse, col = as.factor(model_num))) +
  geom_line(alpha = 0.3) +
  geom_smooth(se = FALSE) +
  labs(col = 'Model', x = 'Epoch', y = 'RMSE') +
  theme_bw()

model %>%
  filter(model_num %in% best_end_modelnum) %>%
  View()
