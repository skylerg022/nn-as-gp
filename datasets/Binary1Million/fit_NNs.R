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
load('data/seed1_split.RData')


# Model Fitting and Potential Picture Generation ------------------------------------

pred <- cbind(x_test,
              latlong_nn_custom = 0, latlong_nn_lee = 0,
              latlong_trans_custom = 0, latlong_trans_lee = 0,
              basis_4by4_custom = 0, basis_4by4_lee = 0,
              basis_4by4_20by20_custom = 0, basis_4by4_20by20_lee = 0)
n_train <- nrow(x_train) + nrow(x_val)

# Lat-Long NN
model_pars_lee <- c(n_layers = 8, layer_width = 2^8, 
                    learning_rate = 0.0001192127,
                    weight_decay = 9.380247e-08, 
                    epochs = 61, batch_size = 2^5, 
                    sigma_w = 1.121034, sigma_b = 0.32748548, 
                    model_num = 100)
model_pars <- c(n_layers = 8, layer_width = 2^8,
                epochs = 45, batch_size = 2^8,
                decay_rate = 0.1 / (n_train %/% 2^8), dropout_rate = 0,
                model_num = 235)

## Lee2018
pred[,'latlong_nn_lee'] <- 
  evalNetwork(x_train, y_train, x_val, y_val,
              model_pars = model_pars_lee, 
              x_test = x_test, y_test = NULL,
              type = 'nn', pars_type = 'lee2018',
              sqrt_n_knots = NULL,
              plot = FALSE)$yhat_test
## Custom
fit <- evalNetwork(x_train, y_train, x_val, y_val,
                   model_pars = model_pars, 
                   x_test = x_test, y_test = NULL,
                   type = 'nn', pars_type = 'custom',
                   sqrt_n_knots = NULL,
                   plot = FALSE)
pred[,'latlong_nn_custom'] <- fit$yhat_test


# Lat-Long Transformed NN
model_pars_lee <- c(n_layers = 4, layer_width = 2^9, 
                    learning_rate = 0.0009401810,
                    weight_decay = 2.496104e-07, 
                    epochs = 55, batch_size = 2^8, 
                    sigma_w = 1.2427385, sigma_b = 0.06185929, 
                    model_num = 7)
model_pars <- c(n_layers = 8, layer_width = 2^7,
                epochs = 60, batch_size = 2^8,
                decay_rate = 0.1 / (n_train %/% 2^8), dropout_rate = 0,
                model_num = 230)

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
                    learning_rate = 0.0024747043,
                    weight_decay = 2.906369e-08, 
                    epochs = 50, batch_size = 2^8, 
                    sigma_w = 2.041133, sigma_b = 0.643556342, 
                    model_num = 90)
model_pars <- c(n_layers = 4, layer_width = 2^9,
                epochs = 95, batch_size = 2^7,
                decay_rate = 0.1 / (n_train %/% 2^7), dropout_rate = 0,
                model_num = 214)

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
# model_pars_lee <- c(n_layers = 4, layer_width = 2^9,
#                     learning_rate = 0.0006105797,
#                     weight_decay = 4.368532e-08,
#                     epochs = 35, batch_size = 2^7,
#                     sigma_w = 0.1983667, sigma_b = 0.11823064,
#                     model_num = 1)
model_pars <- c(n_layers = 4, layer_width = 2^9,
                epochs = 85, batch_size = 2^8,
                decay_rate = 0.1 / (n_train %/% 2^8), dropout_rate = 0,
                model_num = 238)

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
  write_csv('data/dataset2_testpreds.csv')
