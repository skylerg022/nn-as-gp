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
load('data/dataset1_split.RData')


# Model Fitting and Potential Picture Generation ------------------------------------

pred <- cbind(x_test,
              latlong_nn_custom = 0, latlong_nn_lee = 0,
              latlong_trans_custom = 0, latlong_trans_lee = 0,
              basis_4by4_custom = 0, basis_4by4_lee = 0,
              basis_4by4_20by20_custom = 0, basis_4by4_20by20_lee = 0)
n_train <- nrow(x_train) + nrow(x_val)

# Lat-Long NN
model_pars_lee <- c(n_layers = 8, layer_width = 2^7, 
                    learning_rate = 0.0003264859,
                    weight_decay = 1.176411e-07, 
                    epochs = 25, batch_size = 2^6, 
                    sigma_w = 2.457560, sigma_b = 0.3621864, 
                    model_num = 282)
model_pars <- c(n_layers = 8, layer_width = 2^8,
                epochs = 30, batch_size = 2^8,
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
pred[,'latlong_nn_custom'] <- 
  evalNetwork(x_train, y_train, x_val, y_val,
              model_pars = model_pars, 
              x_test = x_test, y_test = NULL,
              type = 'nn', pars_type = 'custom',
              sqrt_n_knots = NULL,
              plot = FALSE)$yhat_test

# Lat-Long Transformed NN
model_pars_lee <- c(n_layers = 4, layer_width = 2^8, 
                    learning_rate = 0.0002608962,
                    weight_decay = 1.145574e-08, 
                    epochs = 35, batch_size = 2^6, 
                    sigma_w = 1.1307929, sigma_b = 0.30043848, 
                    model_num = 226)
model_pars <- c(n_layers = 8, layer_width = 2^8,
                epochs = 30, batch_size = 2^8,
                decay_rate = 0.1 / (n_train %/% 2^8), dropout_rate = 0,
                model_num = 235)

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
                    learning_rate = 0.0003486794,
                    weight_decay = 1.957421e-08, 
                    epochs = 35, batch_size = 2^6, 
                    sigma_w = 2.1830152, sigma_b = 0.52773446, 
                    model_num = 76)
model_pars <- c(n_layers = 8, layer_width = 2^8,
                epochs = 40, batch_size = 2^6,
                decay_rate = 0.1 / (n_train %/% 2^6), dropout_rate = 0,
                model_num = 187)

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
model_pars_lee <- c(n_layers = 4, layer_width = 2^8,
                    learning_rate = 0.0001170766,
                    weight_decay = 1.862151e-08,
                    epochs = 30, batch_size = 2^8,
                    sigma_w = 0.2111619, sigma_b = 0.08122059,
                    model_num = 203)
model_pars <- c(n_layers = 4, layer_width = 2^9,
                epochs = 60, batch_size = 2^7,
                decay_rate = 0.1 / (n_train %/% 2^7), dropout_rate = 0,
                model_num = 214)

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
  write_csv('data/dataset1_testpreds.csv')
