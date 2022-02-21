## Skyler Gray
## Show plots of validation loss performance across 
## neural net grid search

# Load libraries
library(tidyverse)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../HelperFunctions/GridsearchPlots.R')

theme_set(theme_bw())

# Lat-Long NN ----------------------------------------------------------------------

## Lee2018 Gridsearch ------------------------------------------------------

filename1 <- 'data/gridsearch_nn/grid_nn_lee2018'
model <- read_csv(paste0(filename1, '.csv'))
loss <- read_csv(paste0(filename1, '_val_mse.csv')) %>%
  mutate(rmse = sqrt(val_mse)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = TRUE)
View(best5)

## Custom Gridsearch ------------------------------------------------------

filename2 <- 'data/gridsearch_nn/grid_nn'
model <- read_csv(paste0(filename2, '.csv'))
loss <- read_csv(paste0(filename2, '_val_mse.csv')) %>%
  mutate(rmse = sqrt(val_mse)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = FALSE)
View(best5)


# Lat-Long Transformed NN --------------------------------------------------

## Lee2018 Gridsearch ------------------------------------------------------

filename1 <- 'data/gridsearch_nn_trans/grid_nn_trans_lee2018'
model <- read_csv(paste0(filename1, '.csv'))
loss <- read_csv(paste0(filename1, '_val_mse.csv')) %>%
  mutate(rmse = sqrt(val_mse)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = TRUE)
View(best5)

## Custom Gridsearch ------------------------------------------------------

filename2 <- 'data/gridsearch_nn_trans/grid_nn_trans'
model <- read_csv(paste0(filename2, '.csv'))
loss <- read_csv(paste0(filename2, '_val_mse.csv')) %>%
  mutate(rmse = sqrt(val_mse)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = FALSE)
View(best5)


# Basis NN 4by4 ------------------------------------------------------------

## Lee2018 Gridsearch ------------------------------------------------------

filename1 <- 'data/gridsearch_basis/lee2018/grid_basis_4by4'
model <- read_csv(paste0(filename1, '.csv'))
loss <- read_csv(paste0(filename1, '_val_mse.csv')) %>%
  mutate(rmse = sqrt(val_mse)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = TRUE)
View(best5)

## Custom Gridsearch ------------------------------------------------------

filename2 <- 'data/gridsearch_basis/grid_basis_4by4'
model <- read_csv(paste0(filename2, '.csv'))
loss <- read_csv(paste0(filename2, '_val_mse.csv')) %>%
  mutate(rmse = sqrt(val_mse)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = FALSE)
View(best5)


# Basis NN 4by4&20by20 ---------------------------------------------------

## Lee2018 Gridsearch ------------------------------------------------------

filename1 <- 'data/gridsearch_basis/lee2018/grid_basis_4by4_20by20'
model <- read_csv(paste0(filename1, '.csv'))
loss <- read_csv(paste0(filename1, '_val_mse.csv')) %>%
  mutate(rmse = sqrt(val_mse)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = TRUE)
View(best5)

## Custom Gridsearch ------------------------------------------------------

filename2 <- 'data/gridsearch_basis/grid_basis_4by4_20by20'
model <- read_csv(paste0(filename2, '.csv'))
loss <- read_csv(paste0(filename2, '_val_mse.csv')) %>%
  mutate(rmse = sqrt(val_mse)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = FALSE)
View(best5)

