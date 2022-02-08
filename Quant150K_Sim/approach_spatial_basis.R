##
## Spatial Basis Function Expansion
##

## Libraries
library(tidyverse)
library(keras)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../HelperFunctions/MakeNNModel.R')
source('../HelperFunctions/Defaults.R')

# Read in data
load('data/SimulatedTempsSplit.RData')

x_bases <- multiResBases(x_train = x_train,
                         x_test = x_val,
                         sqrt_n_knots = c(4),
                         thresh_type = 'colsum',
                         thresh = 0,
                         thresh_max = 0.75)


# Observe one of the bases
# data_train2 %>%
#   mutate(across(c(x,y), round, digits = 2)) %>%
#   group_by(x, y) %>%
#   summarize(basis = mean(X1)) %>%
#   ggplot(aes(x, y, fill = basis)) +
#   geom_raster()


# Fit a neural network to bases -------------------------------------------

# model_pars <- c(n_layers = 1, layer_width = 2^11,
#                 epochs = 20, batch_size = 2^7,
#                 decay_rate = 0, dropout_rate = 0, 
#                 model_num = 1)
# 
# model <- fitModel(model_pars, x_bases$x_train, y_train,
#                   x_bases$x_test, y_val, test = 'part_train')

model_pars <- c(n_layers = 10, layer_width = 2^8,
                learning_rate = 0.0005590222,
                weight_decay = 1.350768e-06,
                epochs = 20, batch_size = 32,
                sigma_w = 2.232558, sigma_b = 0.153182550,
                model_num = 1250)

model <- fitModelLee2018(model_pars, x_bases$x_train, y_train,
                         x_bases$x_test, y_val, test = 'part_train')


# Predictions after fitting 80% of training set
Predicted <- model %>%
  predict(rbind(x_bases$x_train,
                x_bases$x_test))

data_pred <- cbind(rbind(x_train, x_val),
                   rbind(y_train, y_val), 
                   Predicted) %>%
  # clip predictions to max/min of observed data
  mutate(Predicted = ifelse(Predicted > max(TrueTemp), max(TrueTemp), Predicted),
         Predicted = ifelse(Predicted < min(TrueTemp), min(TrueTemp), Predicted)) %>%
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

x_bases <- multiResBases(x_train = rbind(x_train, x_val),
                         x_test = x_test,
                         sqrt_n_knots = c(4),
                         thresh_type = 'colsum',
                         thresh = 0,
                         thresh_max = 0.75)

model <- fitModelLee2018(model_pars, 
                  x_bases$x_train, 
                  rbind(y_train, y_val), 
                  test = 'all_train')

# Predictions (including test set) after fitting 100% of training set
Predicted <- model %>%
  predict(rbind(x_bases$x_train, x_bases$x_test))

data_pred <- cbind(rbind(x_train, x_val, x_test),
                   rbind(y_train, y_val, y_test), 
                   Predicted) %>%
  pivot_longer(cols = c(TrueTemp, Predicted),
               names_to = 'type',
               values_to = 'z')

simple_train <- data_pred %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y, type) %>%
  summarize(z = mean(z))

# Just test performance
simple_train %>%
  ggplot(aes(x, y, fill = z)) +
  geom_raster() +
  geom_raster(data = cbind(rbind(x_train, x_val),
                           rbind(y_train, y_val)),
              mapping = aes(x, y, fill = min(simple_train$z))) +
  facet_wrap(~type) +
  scale_fill_continuous(type = 'viridis') +
  theme_minimal()

# All data prediction
simple_train %>%
  ggplot(aes(x, y, fill = z)) +
  geom_raster() +
  facet_wrap(~type) +
  scale_fill_continuous(type = 'viridis') +
  theme_minimal()


# Evaluate test RMSE
yhat <- model %>%
  predict(x_bases$x_test)
sqrt(mean( (y_test - yhat)^2 ))

# Evaluate train RMSE
yhat <- model %>%
  predict(x_bases$x_train)
sqrt(mean( (c(y_train, y_val) - yhat)^2 ))

