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
load('data/DataSplit.RData')


# Raw Location NN ----------------------------------------------------------

## Lee2018 Gridsearch ------------------------------------------------------

file_prefix <- 'data/gridsearch/grid_'
filename <- paste0(file_prefix, 'nn_lee2018')
model <- read_csv(paste0(filename, '.csv'))
loss <- read_csv(paste0(filename, '_val_loss.csv')) %>%
  mutate(loss = LossTrans(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = TRUE)
View(best5)

## Custom Gridsearch ------------------------------------------------------

filename <- paste0(file_prefix, 'nn_custom')
model <- read_csv(paste0(filename, '.csv'))
loss <- read_csv(paste0(filename, '_val_loss.csv')) %>%
  mutate(loss = LossTrans(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = FALSE)
View(best5)


# X,Y Transformed NN --------------------------------------------------

## Lee2018 Gridsearch ------------------------------------------------------

filename <- paste0(file_prefix, 'nn_trans_lee2018')
model <- read_csv(paste0(filename, '.csv'))
loss <- read_csv(paste0(filename, '_val_loss.csv')) %>%
  mutate(loss = LossTrans(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = TRUE)
View(best5)

## Custom Gridsearch ------------------------------------------------------

filename <- paste0(file_prefix, 'nn_trans_custom')
model <- read_csv(paste0(filename, '.csv'))
loss <- read_csv(paste0(filename, '_val_loss.csv')) %>%
  mutate(loss = LossTrans(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = FALSE)
View(best5)


# Basis NN 4by4 ------------------------------------------------------------

## Lee2018 Gridsearch ------------------------------------------------------

filename <- paste0(file_prefix, 'basis_4by4_lee2018')
model <- read_csv(paste0(filename, '.csv'))
loss <- read_csv(paste0(filename, '_val_loss.csv')) %>%
  mutate(loss = LossTrans(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = TRUE)
View(best5)

## Custom Gridsearch ------------------------------------------------------

filename <- paste0(file_prefix, 'basis_4by4_custom')
model <- read_csv(paste0(filename, '.csv'))
loss <- read_csv(paste0(filename, '_val_loss.csv')) %>%
  mutate(loss = LossTrans(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = FALSE)
View(best5)


# Basis NN 4by4&20by20 ---------------------------------------------------

## Lee2018 Gridsearch ------------------------------------------------------

filename <- paste0(file_prefix, 'basis_4by4_20by20_lee2018')
model <- read_csv(paste0(filename, '.csv'))
loss <- read_csv(paste0(filename, '_val_loss.csv')) %>%
  mutate(loss = LossTrans(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = TRUE)
View(best5)

## Custom Gridsearch ------------------------------------------------------

filename <- paste0(file_prefix, 'basis_4by4_20by20_custom')
model <- read_csv(paste0(filename, '.csv'))
loss <- read_csv(paste0(filename, '_val_loss.csv')) %>%
  mutate(loss = LossTrans(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = FALSE)
View(best5)

