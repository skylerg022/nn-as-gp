library(tidyverse)
library(keras)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../HelperFunctions/Preprocess.R')
source('../HelperFunctions/NNFunctions.R')
source('../HelperFunctions/Defaults.R')

# Read in data
load('data/dataset2_split.RData')


# Model Fitting and Potential Picture Generation ------------------------------------

pred <- cbind(x_test,
              latlong_nn_custom = 0, latlong_nn_lee = 0,
              latlong_trans_custom = 0, latlong_trans_lee = 0,
              basis_4by4_custom = 0, basis_4by4_lee = 0,
              basis_4by4_20by20_custom = 0, basis_4by4_20by20_lee = 0)
n_train <- nrow(x_train) + nrow(x_val)

# Lat-Long NN
model_pars_lee <- c(n_layers = 8, layer_width = 2^8, 
                    learning_rate = 0.0002030164,
                    weight_decay = 2.646921e-07, 
                    epochs = 30, batch_size = 2^5, 
                    sigma_w = 2.0333117, sigma_b = 0.6507874, 
                    model_num = 253)
model_pars <- c(n_layers = 16, layer_width = 2^7,
                epochs = 90, batch_size = 2^6,
                decay_rate = 0, dropout_rate = 0,
                model_num = 45)

## Lee2018
pred[,'latlong_nn_lee'] <- 
  evalNetwork(x_train, y_train, x_val, y_val,
              model_pars = model_pars_lee, 
              x_test = x_test, y_test = NULL,
              type = 'nn', pars_type = 'lee2018',
              sqrt_n_knots = NULL,
              plot = FALSE)$yhat_test
## Custom
pred[,'latlong_nn_custom'] <- 
  evalNetwork(x_train, y_train, x_val, y_val,
              model_pars = model_pars, 
              x_test = x_test, y_test = NULL,
              type = 'nn', pars_type = 'custom',
              sqrt_n_knots = NULL,
              plot = FALSE)$yhat_test

# Lat-Long Transformed NN
model_pars_lee <- c(n_layers = 4, layer_width = 2^9, 
                    learning_rate = 0.0006413097,
                    weight_decay = 1.650291e-07, 
                    epochs = 35, batch_size = 2^8, 
                    sigma_w = 2.208647, sigma_b = 0.52581911, 
                    model_num = 144)
model_pars <- c(n_layers = 16, layer_width = 2^7,
                epochs = 90, batch_size = 2^6,
                decay_rate = 0, dropout_rate = 0,
                model_num = 40)

## Lee2018
pred[,'latlong_trans_lee'] <- 
  evalNetwork(x_train, y_train, x_val, y_val,
              model_pars = model_pars_lee, 
              x_test = x_test, y_test = NULL,
              type = 'nn_trans', pars_type = 'lee2018',
              sqrt_n_knots = NULL,
              plot = FALSE)$yhat_test
## Custom
pred[,'latlong_trans_custom'] <- 
  evalNetwork(x_train, y_train, x_val, y_val,
              model_pars = model_pars, 
              x_test = x_test, y_test = NULL,
              type = 'nn_trans', pars_type = 'custom',
              sqrt_n_knots = NULL,
              plot = FALSE)$yhat_test


# Basis 4by4
model_pars_lee <- c(n_layers = 8, layer_width = 2^8, 
                    learning_rate = 0.0003339872,
                    weight_decay = 8.919018e-08, 
                    epochs = 35, batch_size = 2^8, 
                    sigma_w = 1.427448, sigma_b = 0.1936748, 
                    model_num = 280)
model_pars <- c(n_layers = 4, layer_width = 2^9,
                epochs = 45, batch_size = 2^7,
                decay_rate = 0.1 / (n_train %/% 2^7), dropout_rate = 0,
                model_num = -1)

## Lee2018
pred[,'basis_4by4_lee'] <- 
  evalNetwork(x_train, y_train, x_val, y_val,
              model_pars = model_pars_lee, 
              x_test = x_test, y_test = NULL,
              type = 'basis', pars_type = 'lee2018',
              sqrt_n_knots = c(4),
              plot = FALSE)$yhat_test
## Custom
pred[,'basis_4by4_custom'] <- 
  evalNetwork(x_train, y_train, x_val, y_val,
              model_pars = model_pars, 
              x_test = x_test, y_test = NULL,
              type = 'basis', pars_type = 'custom',
              sqrt_n_knots = c(4),
              plot = FALSE)$yhat_test


# Basis 4by4&20by20
model_pars_lee <- c(n_layers = 4, layer_width = 2^9,
                    learning_rate = 0.0006105797,
                    weight_decay = 4.368532e-08,
                    epochs = 35, batch_size = 2^7,
                    sigma_w = 0.1983667, sigma_b = 0.11823064,
                    model_num = 1)
model_pars <- c(n_layers = 8, layer_width = 2^7,
                epochs = 20, batch_size = 2^6,
                decay_rate = 0.1 / (n_train %/% 2^6), dropout_rate = 0.1,
                model_num = 70)

## Lee2018
pred[,'basis_4by4_20by20_lee'] <-
  evalNetwork(x_train, y_train, x_val, y_val,
              model_pars = model_pars_lee,
              x_test = x_test, y_test = NULL,
              type = 'basis', pars_type = 'lee2018',
              sqrt_n_knots = c(4, 20),
              plot = FALSE)$yhat_test
## Custom
pred[,'basis_4by4_20by20_custom'] <- 
  evalNetwork(x_train, y_train, x_val, y_val,
              model_pars = model_pars, 
              x_test = x_test, y_test = NULL,
              type = 'basis', pars_type = 'custom',
              sqrt_n_knots = c(4, 20),
              plot = FALSE)$yhat_test

# Save prediction results
pred %>%
  as.data.frame() %>%
  select(-basis_4by4_20by20_lee) %>%
  write_csv('data/quant900k_testpreds.csv')
