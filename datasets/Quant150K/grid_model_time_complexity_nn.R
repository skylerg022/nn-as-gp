library(tidyverse)
library(keras)
library(tictoc)

# set working directory if necessary
tryCatch(setwd('C:/Users/skyle/Documents/GithubRepos/nn-as-gp/Quant150K_Sim'),
         error = function(cond) {
           message(paste0('Could not set directory. ',
                          'Assuming code is being run via Bash.'))
         })
source('../../functions/MakeNNModel.R')

model_grid <- read_csv('data/gridsearch_nn/quant150k_grid.csv')

load('data/AllSimulatedTemps.RData')
data_train <- all.sim.data %>%
  filter(!is.na(MaskTemp))
rm(all.sim.data)

# Only assess model predictive speed using max of 100,000 observations
max_n <- min(nrow(data_train), 1e5)
data_train <- data_train[1:max_n,]


# Neural Network --------------------------------------------------------

x_train <- data_train[,1:2] %>%
  as.matrix.data.frame()
y_train <- data_train[,3, drop = FALSE] %>%
  as.matrix.data.frame()

fastest_model <- function(model_grid) {
  # Only n_layers and layer_width should affect model speed
  n_layers <- min(model_grid$n_layers)
  layer_width <- min(model_grid$layer_width)
  input_shape <- ncol(x_train)
  pars <- c(n_layers, layer_width, 1, 2^7, 0, 0, 0, 1, input_shape)
  
  model <- makeModel(pars)
  model <- model %>%
    compile(loss = 'mse',
            optimizer = optimizer_adam())
  model
}

slowest_model <- function(model_grid) {
  # Only n_layers and layer_width should affect model speed
  n_layers <- max(model_grid$n_layers)
  layer_width <- max(model_grid$layer_width)
  input_shape <- ncol(x_train)
  pars <- c(n_layers, layer_width, 1, 2^7, 0, 0, 0, 1, input_shape)
  
  model <- makeModel(pars)
  model <- model %>%
    compile(loss = 'mse',
            optimizer = optimizer_adam())
  model
}

set.seed(1812)

# prediction speed of whole dataset with fastest model
tic('Fastest Model')
model <- fastest_model(model_grid)
predicted <- model %>%
  predict(x_train)
toc()

# prediction speed of whole dataset with slowest model
tic('Slowest Model')
model <- slowest_model(model_grid)
predicted <- model %>%
  predict(x_train)
toc()
