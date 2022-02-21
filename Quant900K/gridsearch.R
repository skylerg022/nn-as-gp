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

# Create and read in revised data
source('eda.R')
# Remove extra variables and re-source
rm(list = ls())
source('../HelperFunctions/Defaults.R')
source('../HelperFunctions/MakeNNModel.R')
source('../HelperFunctions/GridsearchPlots.R')

# Read in data
load('data/dataset2_split.RData')

n_cores <- 30
gridsearch('nn', n_cores = n_cores)
gridsearch('nn_trans', n_cores = n_cores)

