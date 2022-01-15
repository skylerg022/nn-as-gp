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
source('../HelperFunctions/MakeNNModel.R')
source('../HelperFunctions/Defaults.R')

# Read in data
load('data/AllSimulatedTemps.RData')
data_train <- all.sim.data %>%
  rename(x = Lon, y = Lat) %>%
  filter(!is.na(MaskTemp)) %>%
  # Make validation set: Because blocks of data are in test dataset, 
  #  validation set should be blocks
  mutate(validation = ifelse( (((x > -94.5 & x < -93) & (y < 34.75)) |
                                 ((x > -93.5 & x < -93) & (y > 35.5 & y < 36)) |
                                 ((x > -91.75) & (y > 35.25 & y < 35.75)) |
                                 ((x > -95.75 & x < -95.25) & (y > 36.25 & y < 36.75)) |
                                 ((x > -95 & x < -94.5) & (y > 35.25 & y < 35.75)) |
                                 ((x > -92.5 & x < -91.75) & (y > 36 & y < 36.75))),
                              1, 0)) %>%
  arrange(validation)

rm(all.sim.data)


# Neural Network --------------------------------------------------------

VAL_SPLIT <- 0.2
x_train <- cbind(data_train[,1:2], data_train[,1:2]^2) %>%
  # Center and scale using training data, not validation or test
  predictorsScaled(val_split = VAL_SPLIT)

y_train <- data_train[,3, drop = FALSE] %>%
  as.matrix()

# Set constants
TRAIN_SIZE <- (1 - VAL_SPLIT) * nrow(x_train)

grid <- expand.grid(n_layers = c(1, 2, 4, 8, 16), 
                    layer_width = c(2^3, 2^6, 2^7),
                    epochs = c(20, 50, 100), 
                    batch_size = c(2^6, 2^7, 2^8),
                    decay_rate = c(0, 0.05),
                    dropout_rate = c(0, 0.5)) %>%
  mutate(decay_rate = decay_rate / 
           (TRAIN_SIZE %/% batch_size),
         model_num = 1:n())

# Make grid into input class: list
grid_list <- split(grid, 1:nrow(grid))

# NORMAL TIME TO PROCESS ONE OF THE NN PARAMETER SETS
# time_unicore <- system.time({
#   set.seed(1812)
#   lapply(grid_list, fitModel)
# })

# Use at most the number of cores available on server
n_cores <- 20
time <- system.time({
  results <- mclapply(grid_list, 
                      function(pars) fitModel(pars, x_train, y_train, test = 'grid'),
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
write_csv(grid, 'data/quant150k_grid_nn.csv')
write_csv(val_df, 'data/quant150k_grid_nn_val_mse.csv')

