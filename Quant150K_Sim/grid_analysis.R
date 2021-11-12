## Skyler Gray
## Show plots of validation loss performance across 
## neural net grid search

# Load libraries
library(tidyverse)

# set working directory if necessary
tryCatch(setwd('C:/Users/skyle/Documents/GithubRepos/nn-as-gp/Quant150K_Sim'),
         error = function(cond) {
           message(paste0('Could not set directory. ',
                          'Assuming code is being run via Bash.'))
         })

model <- read_csv('data/gridsearch_nn/quant150k_grid.csv')
loss <- read_csv('data/gridsearch_nn/quant150k_grid_val_mse.csv') %>%
  mutate(rmse = sqrt(val_mse)) %>%
  inner_join(model, by = 'model_num')

# Grid Search: General Trends ---------------------------------------------

# Filter down to RMSE less than 5
loss %>%
  ggplot(aes(x = epoch, y = rmse, group = model_num)) +
  geom_smooth(se = FALSE, alpha = 0.5) +
  facet_grid(epochs ~ layer_width) +
  scale_y_continuous(limits = c(NA, 5))
# Seems like 30 epochs is not sufficient to get good validation performance.
# Consider removing 30 epochs option and adding 100 epoch option.

loss %>%
  ggplot(aes(x = epoch, y = rmse, group = model_num)) +
  geom_smooth(se = FALSE, alpha = 0.5, col = 'royalblue') +
  facet_grid(epochs ~ decay_rate) +
  scale_y_continuous(limits = c(NA, 5))
# Current decay rate settings may be too fast
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
           batch_size, decay_rate, square_feat, dropout_rate) %>%
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
