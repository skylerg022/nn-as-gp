## Gridsearch algorithm to tune neural network hyperparameters
## for the simulated satellite data.

## This is a gridsearch algorithm implemented for use through
## either MacOS or some Linux OS. This implementation will not
## run with parallelization on Windows OS

library(tidyverse)
library(keras)
library(parallel)

# Helper functions and data ------------------------------------------------------------

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}
source('eda.R')
source('../HelperFunctions/MakeNNModel.R')
source('../HelperFunctions/Defaults.R')

# Make directories if needed
dirCheck()

# Read in data
load('data/SimulatedTempsSplit.RData')

gridsearch <- function(type = 'nn', n_cores = 20) {
  if (type == 'nn') {
    # x_train and x_val are ready for scaling
  } else if (type == 'nn_trans') {
    x_train <- cbind(x_train, x_train^2)
    x_val <- cbind(x_val, x_val^2)
  } else {
    err_message <- paste0('Grid search not recognized. Please assign type ',
                          'as "nn" or "nn_trans".')
    stop(err_message)
  }
  
  # Neural Network --------------------------------------------------------
  
  n_train <- nrow(x_train)
  # Center and scale train and val using training data only
  x_scaled <- predictorsScaled(x_train, x_val)
  x_train <- x_scaled[1:n_train,]
  x_val <- x_scaled[-c(1:n_train),]
  
  grid <- expand.grid(n_layers = c(1, 2, 4, 8, 16), 
                      layer_width = c(2^3, 2^6, 2^7),
                      epochs = c(20, 50, 100), 
                      batch_size = c(2^6, 2^7, 2^8),
                      decay_rate = c(0, 0.05),
                      dropout_rate = c(0, 0.5)) %>%
    mutate(decay_rate = decay_rate / 
             (n_train %/% batch_size),
           model_num = 1:n())
  
  # Make grid into input class: list
  grid_list <- split(grid, 1:nrow(grid))
  
  # NORMAL TIME TO PROCESS ONE OF THE NN PARAMETER SETS
  # time_unicore <- system.time({
  #   set.seed(1812)
  #   lapply(grid_list, fitModel)
  # })
  
  # Use at most the number of cores available on server
  time <- system.time({
    results <- mclapply(grid_list, 
                        function(pars) fitModel(pars, x_train, y_train, 
                                                x_val, y_val, test = 'grid'),
                        mc.cores = n_cores, mc.silent = FALSE)
                        # mc.cleanup = FALSE, mc.allow.recursive = FALSE)
  })
  
  ## BEGINNING OF LOG CODE
  # fileConn <- file("log_gridfit.txt")
  # open(fileConn, open = 'a')
  # writeLines(c("Hello","World"), fileConn)
  # close(fileConn)
  
  # Convert results from list into dataframe
  val_df <- matrix(NA, nrow = 0, ncol = 3,
                   dimnames = list(NULL, c('model_num', 'epoch', 'val_mse')))
  for (i in 1:length(results)) {
    epochs <- length(results[[i]]$val_loss)
    new_rows <- data.frame(model_num = results[[i]]$pars[['model_num']],
                           epoch = 1:epochs,
                           val_mse = results[[i]]$val_loss)
    val_df <- rbind(val_df, new_rows)
  }
  
  # Write data to csv
  write_csv(grid, paste0('data/quant150k_grid_', type, '.csv'))
  write_csv(val_df, paste0('data/quant150k_grid_', type, '_val_mse.csv'))
  
  return()
}

n_cores <- 30
gridsearch('nn', n_cores = n_cores)
gridsearch('nn_trans', n_cores = n_cores)

