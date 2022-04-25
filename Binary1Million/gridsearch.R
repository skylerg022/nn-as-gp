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

# Make directories if needed
source('../HelperFunctions/Defaults.R')
dirCheck()
source('../HelperFunctions/NNFunctions.R')
source('../HelperFunctions/Preprocess.R')
source('../HelperFunctions/Gridsearch.R')


# Read in data
load('data/seed1_split.RData')

n_cores <- 30
gridsearch(input_type = 'nn', modeltype = 'custom', 
           n_cores = n_cores, sqrt_n_knots = NULL, 
           binary_data = TRUE,
           loss = loss_binary_crossentropy())
gridsearch(input_type = 'nn', modeltype = 'lee2018', 
           n_cores = n_cores, sqrt_n_knots = NULL,
           binary_data = TRUE,
           loss = loss_binary_crossentropy())
