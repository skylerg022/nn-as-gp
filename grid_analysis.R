## Produce plots of validation loss performance across 
##  neural net grid search; save best performing network
##  hyperparameter settings for all 8 models (raw, 
##  transformed, 4x4 Basis, and Multi-Resolution Basis
##  for both Lee2018 and Custom grid searches)

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
dir_name <- paste0(getwd(), '/datasets/', args[1])
if (!dir.exists(dir_name)) {
  message <- paste0('Directory ', dir_name, ' does not exist.')
  stop(message)
}
setwd(dir_name)

# If second argument is FALSE, validation loss is MSE and must 
#  be transformed to RMSE
binary_data <- as.logical(args[2])
if (binary_data == TRUE) {
  LossTrans <- function(x) x
} else {
  LossTrans <- function(x) sqrt(x)
}

source('../../functions/Defaults.R')
source('../../functions/Gridsearch.R')

# Create needed directories for plots
CheckDir()

# Read in data
# load('data/DataSplit.RData')


mod_types <- expand.grid(pars_type = c('custom', 'lee2018'),
                         type = c('nn', 'nn_trans', 'basis_4by4', 'basis_4by4_20by20'))

# Load grid search result csv's; identify and return best model;
#   also save plot summarizing best loss at different layer-width settings
models <- apply(mod_types, 1,
                function(mod_type) {
                  type <- mod_type[2]
                  pars_type <- mod_type[1]
                  file_suffix <- paste0(type, '_', pars_type)
                  filename <- paste0(file_prefix, file_suffix)
                  
                  model <- read_csv(paste0(filename, '.csv')) %>%
                    suppressMessages()
                  loss <- read_csv(paste0(filename, '_val_loss.csv')) %>%
                    suppressMessages() %>%
                    mutate(loss = LossTrans(val_loss)) %>%
                    inner_join(model, by = 'model_num')
                  
                  best <- gridsearchEDAandClean(model, loss, pars_type = pars_type,
                                                binary_data = binary_data,
                                                filename = file_suffix)
                  best
                }) %>%
  reduce(bind_rows)

# Save best models in preparation for fitting them
bind_cols(mod_types, models) %>%
  write_csv('data/final_models_setup.csv')
