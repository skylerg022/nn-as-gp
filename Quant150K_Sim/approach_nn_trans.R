library(tidyverse)
library(keras)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}
source('../HelperFunctions/MakeNNModel.R')
source('../HelperFunctions/Defaults.R')

# Read in data
load('data/AllSimulatedTemps.RData')
data_train <- all.sim.data %>%
  rename(x = Lon, y = Lat) %>%
  filter(!is.na(MaskTemp)) %>%
  # Because blocks of data are in test dataset, validation set should be blocks
  mutate(validation = ifelse( (((x > -94.5 & x < -93) & (y < 34.75)) |
                                 ((x > -93.5 & x < -93) & (y > 35.5 & y < 36)) |
                                 ((x > -91.75) & (y > 35.25 & y < 35.75)) |
                                 ((x > -95.75 & x < -95.25) & (y > 36.25 & y < 36.75)) |
                                 ((x > -95 & x < -94.5) & (y > 35.25 & y < 35.75)) |
                                 ((x > -92.5 & x < -91.75) & (y > 36 & y < 36.75))),
                              1, 0)) %>%
  arrange(validation)
data_test <- all.sim.data %>%
  rename(x = Lon, y = Lat) %>%
  filter(is.na(MaskTemp))
rm(all.sim.data)


# Neural Network --------------------------------------------------------

# Train on 80% of the training data and validate
VAL_SPLIT <- 0.2
x_train <- cbind(data_train[,1:2], data_train[,1:2]^2)
  # Center and scale using training data, not validation or test
  predictorsScaled(val_split = VAL_SPLIT)

y_train <- data_train[,3, drop = FALSE] %>%
  as.matrix()

# One of the top five performers in sim_satellite nn grid search (1/15/2022)
model_pars <- c(n_layers = 16, layer_width = 2^6,
                epochs = 50, batch_size = 2^6,
                decay_rate = 0, dropout_rate = 0, 
                model_num = 25)

model <- fitModel(model_pars, x_train, y_train, test = 'part_train')

# What areas are solely validation data?
# data_train %>%
#   mutate(across(c(x,y), round, digits = 2),
#          training = ifelse(row_number()/n() < 0.8, 1, 0)) %>%
#   group_by(x, y) %>%
#   summarize(training = min(training)) %>%
#   mutate(Dataset = ifelse(training == 1, 'Training', 'Validation')) %>%
#   ggplot(aes(x, y, fill = Dataset)) +
#   geom_raster() +
#   scale_fill_brewer(type = 'qual', palette = 6)


# Predictions after fitting 80% of training set
Predicted <- model %>%
  predict(x_train)

data_pred <- cbind(data_train, Predicted) %>%
  # clip predictions to max/min of observed data
  # mutate(Predicted = ifelse(Predicted > max(TrueTemp), max(TrueTemp), Predicted),
  #        Predicted = ifelse(Predicted < min(TrueTemp), min(TrueTemp), Predicted)) %>%
  pivot_longer(cols = c(TrueTemp, Predicted),
               names_to = 'type',
               values_to = 'z')

simple_train <- data_pred %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y, type) %>%
  summarize(Temp = mean(z))

simple_train %>%
  ggplot(aes(x, y, fill = Temp)) +
  geom_raster() +
  facet_wrap(~type) +
  scale_fill_continuous(type = 'viridis') +
  theme_minimal()


# Test Predictions --------------------------------------------------------

n <- nrow(data_train)
x <- predictorsScaled(cbind(data_train[,1:2], data_train[,1:2]^2), 
                      cbind(data_test[,1:2], data_test[,1:2]^2), 
                      val_split = 0)
x_train <- x[1:n,]
x_test <- x[-c(1:n),]
rm(x)

y_train <- data_train[,3, drop = FALSE] %>%
  as.matrix()
y_test <- data_test[,4, drop = FALSE] %>%
  as.matrix()

model <- fitModel(model_pars, x_train, y_train, test = 'all_train')

# Predictions (including test set) after fitting 100% of training set
Predicted <- model %>%
  predict(rbind(x_train, x_test))

data_pred <- cbind(rbind(data_train %>% select(-validation), 
                         data_test), 
                   Predicted) %>%
  pivot_longer(cols = c(TrueTemp, Predicted),
               names_to = 'type',
               values_to = 'z')

simple_train <- data_pred %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y, type) %>%
  summarize(z = mean(z))

# Just test performance
p_test <- simple_train %>%
  ggplot(aes(x, y, fill = z)) +
  geom_raster() +
  geom_raster(data = data_train,
              mapping = aes(x, y, fill = min(simple_train$z))) +
  facet_wrap(~type) +
  scale_fill_continuous(type = 'viridis') +
  theme_minimal() +
  labs(fill = 'Temp')

# All data prediction
p_all <- simple_train %>%
  ggplot(aes(x, y, fill = z)) +
  geom_raster() +
  facet_wrap(~type) +
  scale_fill_continuous(type = 'viridis') +
  theme_minimal() +
  labs(fill = 'Temp')

# Evaluate test RMSE
yhat <- model %>%
  predict(x_test)
rmse_test <- sqrt(mean( (y_test - yhat)^2 )) %>%
  round(2)

# Evaluate train RMSE
yhat <- model %>%
  predict(x_train)
rmse_train <- sqrt(mean( (y_train - yhat)^2 )) %>%
  round(2)


# Save ggplots
filename <- paste0('pics/nn_rmsetrain', rmse_train,
                   '_test', rmse_test, '_showtest.png') %>%
  str_replace_all('(?<=[0-9])\\.(?=[0-9])', '_')
ggsave(filename,
       plot = p_test,
       width = pic_width,
       height = pic_height,
       units = pic_units,
       bg = 'white')

filename <- paste0('pics/nn_rmsetrain', rmse_train,
                   '_test', rmse_test, '_showall.png') %>%
  str_replace_all('(?<=[0-9])\\.(?=[0-9])', '_')
ggsave(filename,
       plot = p_all,
       width = pic_width,
       height = pic_height,
       units = pic_units,
       bg = 'white')
