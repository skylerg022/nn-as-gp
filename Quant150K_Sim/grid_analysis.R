## Skyler Gray
## Show plots of validation loss performance across 
## neural net grid search

# Load libraries
library(tidyverse)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

filename1 <- 'data/gridsearch_basis/lee2018/grid_basis_4by4_20by20'
model <- read_csv(paste0(filename1, '.csv'))
loss <- read_csv(paste0(filename1, '_val_mse.csv')) %>%
  mutate(rmse = sqrt(val_mse)) %>%
  inner_join(model, by = 'model_num')

# Grid Search: General Trends ---------------------------------------------

# Filter down to RMSE less than 5
loss %>%
  ggplot(aes(x = epoch, y = rmse, group = model_num)) +
  geom_smooth(se = FALSE, alpha = 0.5) +
  facet_grid(n_layers ~ layer_width) 
  scale_y_continuous(limits = c(NA, 5))

loss %>%
  ggplot(aes(x = epoch, y = rmse, group = model_num)) +
  geom_smooth(se = FALSE, alpha = 0.5, col = 'royalblue') +
  facet_grid(epochs ~ decay_rate) +
  scale_y_continuous(limits = c(NA, 5))


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
  geom_line(alpha = 0.3) +
  geom_smooth(se = FALSE) +
  labs(col = 'Model', x = 'Epoch', y = 'RMSE') +
  theme_bw()

loss_best %>%
  group_by(model_num) %>%
  summarize(min_rmse = round(min(rmse), 2),
            min_rmse_epoch = which.min(rmse)) %>%
  inner_join(model, by = 'model_num') %>%
  select(-c(min_rmse, min_rmse_epoch),
         c(min_rmse, min_rmse_epoch)) %>%
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
