## Gridsearch algorithm to tune basis neural network hyperparameters
## for the simulated satellite data.

## This is a gridsearch algorithm implemented for use through
## either MacOS or some Linux OS. This implementation will not
## run with parallelization on Windows OS

library(tidyverse)
library(keras)
library(parallel)

# Set seed
set.seed(2422)

# Helper functions and data ------------------------------------------------------------

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

# Load functions and make directories if needed
source('../HelperFunctions/Defaults.R')
source('../HelperFunctions/MakeNNModel.R')
dirCheck()

# Create and read in revised data
source('eda.R')

# Read in data
load('data/SimulatedTempsSplit.RData')

gridsearch <- function(modeltype = 'custom', n_cores = 20) {
  
  # Neural Network --------------------------------------------------------
  
  n_train <- nrow(x_train)
  
  # Center and scale train and val using training data only
  # x_scaled <- predictorsScaled(x_train, x_val)
  # x_train <- x_scaled[1:n_train,]
  # x_val <- x_scaled[-c(1:n_train),]
  
  if (modeltype == 'custom') {
    grid <- expand.grid(n_layers = c(1, 2, 4, 8, 16),
                        layer_width = c(2^3, 2^7, 2^8, 2^9, 2^10),
                        epochs = 30,
                        batch_size = c(2^6, 2^7, 2^8),
                        decay_rate = c(0, 0.05),
                        dropout_rate = c(0, 0.1)) %>%
      filter(layer_width^2 * (n_layers-1) < 800000) %>%
      arrange(desc(layer_width^2 * (n_layers-1))) %>% # Largest to smallest models
      mutate(decay_rate = decay_rate /
               (n_train %/% batch_size),
             model_num = row_number())
  } else if (modeltype == 'lee2018') {
    grid <- makeGridLee2018()
  }
  
  # Make grid into input class: list
  grid_list <- split(grid, 1:nrow(grid))
  
  # Use at most the number of cores available on server
  time <- system.time({
    results <- mclapply(grid_list, 
                        function(pars) {
                          x_bases <- multiResBases(x_train = x_train,
                                                   x_test = x_val,
                                                   sqrt_n_knots = c(4),
                                                   thresh_type = 'colsum',
                                                   thresh = 0,
                                                   thresh_max = 0)
                          
                          fitModel(pars, x_bases$x_train, y_train, 
                                   x_bases$x_test, y_val,
                                   modeltype = modeltype,
                                   test = 'grid') 
                          },
                        mc.cores = n_cores, mc.silent = FALSE)
                        # mc.cleanup = FALSE, mc.allow.recursive = FALSE)
  })
  
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
  write_csv(grid, paste0('data/grid_basis_', modeltype, '.csv'))
  write_csv(val_df, paste0('data/grid_basis_', modeltype, '_val_mse.csv'))
  
  return()
}

n_cores <- 30
gridsearch(modeltype = 'custom', n_cores = n_cores)

