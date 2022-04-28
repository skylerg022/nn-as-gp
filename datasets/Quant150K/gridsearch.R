## Gridsearch algorithm to tune neural network hyperparameters
## for the simulated satellite data.

## This is a gridsearch algorithm implemented for use through
## either MacOS or some Linux OS. This implementation will not
## run with parallelization on Windows OS


# Helper functions and data ------------------------------------------------------------

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

# Load libraries and needed functions
source('../../functions/Defaults.R')
source('../../functions/NNFunctions.R')
source('../../functions/Preprocess.R')
source('../../functions/Gridsearch.R')

# Read in data
load('data/SatelliteTempsSplit.RData')

n_cores <- 30
# Untransformed inputs
gridsearch(input_type = 'nn', modeltype = 'custom', 
           n_cores = n_cores, sqrt_n_knots = NULL,
           loss = loss_mean_squared_error())
gridsearch(input_type = 'nn', modeltype = 'lee2018', 
           n_cores = n_cores, sqrt_n_knots = NULL,
           loss = loss_mean_squared_error())
# Transformed inputs
gridsearch(input_type = 'nn_trans', modeltype = 'custom', 
           n_cores = n_cores, sqrt_n_knots = NULL,
           loss = loss_mean_squared_error())
gridsearch(input_type = 'nn_trans', modeltype = 'lee2018', 
           n_cores = n_cores, sqrt_n_knots = NULL,
           loss = loss_mean_squared_error())
# Radial basis function expansion
gridsearch(input_type = 'basis', modeltype = 'custom', 
           n_cores = n_cores, sqrt_n_knots = c(4),
           loss = loss_mean_squared_error())
gridsearch(input_type = 'basis', modeltype = 'lee2018', 
           n_cores = n_cores, sqrt_n_knots = c(4),
           loss = loss_mean_squared_error())
# Multi-resolution basis function expansion
gridsearch(input_type = 'basis', modeltype = 'custom', 
           n_cores = n_cores, sqrt_n_knots = c(4,20),
           loss = loss_mean_squared_error())
gridsearch(input_type = 'basis', modeltype = 'lee2018', 
           n_cores = n_cores, sqrt_n_knots = c(4,20),
           loss = loss_mean_squared_error())
