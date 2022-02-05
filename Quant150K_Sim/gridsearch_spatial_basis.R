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
source('eda.R')
rm(list = ls())
source('../HelperFunctions/MakeNNModel.R')
source('../HelperFunctions/Defaults.R')

# Make directories if needed
dirCheck()

# Read in data
load('data/SimulatedTempsSplit.RData')

gridsearch <- function(n_cores = 20) {
  
  # Neural Network --------------------------------------------------------
  
  n_train <- nrow(x_train)
  
  # Center and scale train and val using training data only
  # x_scaled <- predictorsScaled(x_train, x_val)
  # x_train <- x_scaled[1:n_train,]
  # x_val <- x_scaled[-c(1:n_train),]
  
  # grid <- expand.grid(n_layers = 1,
  #                     layer_width = 2^7,
  #                     epochs = 20, 
  #                     batch_size = 2^7,
  #                     decay_rate = 0,
  #                     dropout_rate = 0,
  #                     thresh_colsums = seq(0, 100, by = 10),
  #                     thresh_max = seq(0, 0.95, by = 0.05)) %>%
  #   mutate(decay_rate = decay_rate / 
  #            (n_train %/% batch_size),
  #          model_num = 1:n())
  
  grid <- makeGridLee2018()
  
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
                        function(pars) {
                          x_bases <- multiResBases(x_train = x_train,
                                                   x_test = x_val,
                                                   sqrt_n_knots = c(4),
                                                   thresh_type = 'colsum',
                                                   thresh = 0,
                                                   thresh_max = 0)
                          
                          fitModelLee2018(pars, x_bases$x_train, y_train, 
                                          x_bases$x_test, y_val, test = 'grid') 
                          },
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
  write_csv(grid, paste0('data/quant150k_grid_basis.csv'))
  write_csv(val_df, paste0('data/quant150k_grid_basis_val_mse.csv'))
  
  return()
}

n_cores <- 30
gridsearch(n_cores = n_cores)

