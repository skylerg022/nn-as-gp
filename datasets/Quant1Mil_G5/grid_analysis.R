## Skyler Gray
## Show plots of validation loss performance across 
## neural net grid search

# Load libraries
library(tidyverse)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../../functions/Gridsearch.R')

theme_set(theme_bw())

# X,Y NN ----------------------------------------------------------------------

## Lee2018 Gridsearch ------------------------------------------------------

filename1 <- 'data/dataset1/grid_nn_lee2018'
model <- read_csv(paste0(filename1, '.csv'))
loss <- read_csv(paste0(filename1, '_val_loss.csv')) %>%
  mutate(loss = sqrt(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = TRUE)
View(best5)

## Custom Gridsearch ------------------------------------------------------

filename2 <- 'data/dataset1/grid_nn_custom'
model <- read_csv(paste0(filename2, '.csv'))
loss <- read_csv(paste0(filename2, '_val_loss.csv')) %>%
  mutate(loss = sqrt(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = FALSE)
View(best5)


# X,Y Transformed NN --------------------------------------------------

## Lee2018 Gridsearch ------------------------------------------------------

filename1 <- 'data/dataset1/grid_nn_trans_lee2018'
model <- read_csv(paste0(filename1, '.csv'))
loss <- read_csv(paste0(filename1, '_val_loss.csv')) %>%
  mutate(loss = sqrt(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = TRUE)
View(best5)

## Custom Gridsearch ------------------------------------------------------

filename2 <- 'data/dataset1/grid_nn_trans_custom'
model <- read_csv(paste0(filename2, '.csv'))
loss <- read_csv(paste0(filename2, '_val_loss.csv')) %>%
  mutate(loss = sqrt(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = FALSE)
View(best5)


# Basis NN 4by4 ------------------------------------------------------------

## Lee2018 Gridsearch ------------------------------------------------------

filename1 <- 'data/dataset1/grid_basis_4by4_lee2018'
model <- read_csv(paste0(filename1, '.csv'))
loss <- read_csv(paste0(filename1, '_val_loss.csv')) %>%
  mutate(loss = sqrt(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = TRUE)
View(best5)

## Custom Gridsearch ------------------------------------------------------

filename2 <- 'data/dataset1/grid_basis_4by4_custom'
model <- read_csv(paste0(filename2, '.csv'))
loss <- read_csv(paste0(filename2, '_val_loss.csv')) %>%
  mutate(loss = sqrt(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = FALSE)
View(best5)


# Basis NN 4by4&20by20 ---------------------------------------------------

## Lee2018 Gridsearch ------------------------------------------------------

filename1 <- 'data/dataset1/grid_basis_4by4_20by20_lee2018'
model <- read_csv(paste0(filename1, '.csv'))
loss <- read_csv(paste0(filename1, '_val_loss.csv')) %>%
  mutate(loss = sqrt(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = TRUE)
View(best5)

## Custom Gridsearch ------------------------------------------------------

filename2 <- 'data/dataset1/grid_basis_4by4_20by20_custom'
model <- read_csv(paste0(filename2, '.csv'))
loss <- read_csv(paste0(filename2, '_val_loss.csv')) %>%
  mutate(loss = sqrt(val_loss)) %>%
  inner_join(model, by = 'model_num')

best5 <- gridsearchEDAandClean(model, loss, lee2018 = FALSE)
View(best5)

