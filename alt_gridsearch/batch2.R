## Gridsearch algorithm to tune neural network hyperparameters
## for prepared data saved in the data/DataSplit.RData file

## This is a gridsearch algorithm implemented for use through
## either MacOS or some Linux OS. This implementation will not
## run with parallelization on Windows OS

# This code is meant to be run via the bash command line using
#  Rscript fit_NNs.R <dataset name> <binary_data>
#  where <dataset name> is a string for the name of the directory holding the
#  eda.R and data files of a given dataset and <binary_data> is either TRUE or
#  FALSE to indicate whether the response data is binary or continuous
args <- commandArgs(trailingOnly = TRUE)

# ...however, uncomment the following lines to run this code in RStudio
# Set working directory if using RStudio
# if (rstudioapi::isAvailable()) {
#   setwd(dirname(rstudioapi::getSourceEditorContext()$path))
# }
# args <- c('', 'FALSE')

# Check that dataset exists
dir_name <- paste0(getwd(), '../datasets/', args[1])
if (!dir.exists(dir_name)) {
  message <- paste0('Directory ', dir_name, ' does not exist.')
  stop(message)
}
setwd(dir_name)

# Helper functions
# Make directories if needed
source('../../functions/Defaults.R')
CheckDir()
source('../../functions/NNFunctions.R')
source('../../functions/Preprocess.R')
source('../../functions/Gridsearch.R')

# Read in data
load('data/DataSplit.RData')

# If second argument is FALSE, validation loss is MSE and must 
#  be transformed to RMSE
binary_data <- as.logical(args[2])
if (binary_data == TRUE) {
  loss <- loss_binary_crossentropy()
} else {
  loss <- loss_mean_squared_error()
}


# Grid searches
# NOTE: These grid searches are large and may take a day EACH to finish with
#   the default ncore settings shown below.

# Transformed location input
n_cores <- 30
gridsearch(input_type = 'nn_trans', modeltype = 'custom', 
           n_cores = n_cores, sqrt_n_knots = NULL, 
           binary_data = binary_data,
           loss = loss)
gridsearch(input_type = 'nn_trans', modeltype = 'lee2018', 
           n_cores = n_cores, sqrt_n_knots = NULL,
           binary_data = binary_data,
           loss = loss)

