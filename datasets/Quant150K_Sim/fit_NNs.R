library(tidyverse)
library(keras)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../../functions/Preprocess.R')
source('../../functions/NNFunctions.R')
source('../../functions/Defaults.R')

# Read in data
load('data/SimulatedTempsSplit.RData')


# Model Fitting and Potential Picture Generation ------------------------------------

pred <- cbind(x_test,
              latlong_nn_custom = 0, latlong_nn_lee = 0,
              latlong_trans_custom = 0, latlong_trans_lee = 0,
              basis_4by4_custom = 0, basis_4by4_lee = 0,
              basis_4by4_20by20_custom = 0, basis_4by4_20by20_lee = 0)
n_train <- nrow(x_train) + nrow(x_val)

# Lat-Long NN
model_pars_lee <- c(n_layers = 4, layer_width = 2^6, 
                    learning_rate = 0.0030813342,
                    weight_decay = 9.079531e-05, 
                    epochs = 15, batch_size = 2^5, 
                    sigma_w = 0.1286637, sigma_b = 0.4485857, 
                    model_num = 592)
model_pars <- c(n_layers = 8, layer_width = 2^8,
                epochs = 15, batch_size = 2^5,
                decay_rate = 0, dropout_rate = 0,
                model_num = 43)

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
model_pars_lee <- c(n_layers = 8, layer_width = 2^6, 
                    learning_rate = 0.0034654830,
                    weight_decay = 1.013484e-08, 
                    epochs = 15, batch_size = 2^5, 
                    sigma_w = 1.782659, sigma_b = 1.4626552, 
                    model_num = 472)
model_pars <- c(n_layers = 16, layer_width = 2^7,
                epochs = 10, batch_size = 2^4,
                decay_rate = 0, dropout_rate = 0,
                model_num = 15)

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
model_pars_lee <- c(n_layers = 16, layer_width = 2^6, 
                    learning_rate = 0.0015395834,
                    weight_decay = 3.216239e-08, 
                    epochs = 20, batch_size = 2^4, 
                    sigma_w = 0.8315338, sigma_b = 0.3250620, 
                    model_num = 378)
model_pars <- c(n_layers = 4, layer_width = 2^9,
                epochs = 20, batch_size = 2^6,
                decay_rate = 0, dropout_rate = 0.1,
                model_num = 310)

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
model_pars_lee <- c(n_layers = 4, layer_width = 2^7,
                    learning_rate = 0.0012925395,
                    weight_decay = 4.670805e-07,
                    epochs = 5, batch_size = 2^7,
                    sigma_w = 0.9120156, sigma_b = 0.3925908,
                    model_num = 428)
model_pars <- c(n_layers = 4, layer_width = 2^8,
                epochs = 15, batch_size = 2^8,
                decay_rate = 0.1 / (n_train %/% 2^8), dropout_rate = 0,
                model_num = 138)

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
  write_csv('data/quant150ksim_testpreds.csv')
