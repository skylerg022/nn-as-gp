library(tidyverse)
library(keras)
library(parallel)

# Read in Data ------------------------------------------------------------

# set working directory if necessary
tryCatch(setwd('C:/Users/skyle/Documents/GithubRepos/nn-as-gp/Quant150K_Sim'),
         error = function(cond) {
           message(paste0('Could not set directory. ',
                          'Assuming code is being run via Bash.'))
         })
source('../HelperFunctions/MakeNNModel.R')

# Read in data
load('data/AllSimulatedTemps.RData')
data_train <- all.sim.data %>%
  rename(x = Lon, y = Lat) %>%
  filter(!is.na(MaskTemp)) %>%
  # Because blocks of data are in test dataset, validation set should be blocks
  mutate(validation = ifelse( (((x > -94.5 & x < -93) & (y < 34.75)) |
                                 ((x > -93.5 & x < -93) & (y > 35.5 & y < 36)) |
                                 ((x > -91.75) & (y > 35.25 & y < 35.75)) |
                                 ((x > -95.75 & x < -95.25) & (y > 36.25 & y < 36.75)) |
                                 ((x > -95 & x < -94.5) & (y > 35.25 & y < 35.75)) |
                                 ((x > -92.5 & x < -91.75) & (y > 36 & y < 36.75))),
                              1, 0)) %>%
  arrange(validation)
data_test <- all.sim.data %>%
  rename(x = Lon, y = Lat) %>%
  filter(is.na(MaskTemp))
rm(all.sim.data)


# Neural Network --------------------------------------------------------

x_train <<- data_train[,1:2] %>%
  as.matrix()
y_train <<- data_train[,3, drop = FALSE] %>%
  as.matrix()

VAL_SPLIT <- 0.2
TRAIN_SIZE <- (1 - VAL_SPLIT) * nrow(x_train)

grid <- expand.grid(n_layers = c(1, 2, 4, 8, 16), 
                    layer_width = c(2^3, 2^6, 2^7),
                    epochs = c(20, 50, 100), 
                    batch_size = c(2^6, 2^7, 2^8),
                    decay_rate = c(0, 0.05),
                    square_feat = c(0, 1),
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

# Rencher has 40 total cores. Use 20 of those 40
n_cores <- 20
time <- system.time({
  results <- mclapply(grid_list, 
                      function(pars) fitModel(pars, x_train, y_train, test = 'part_train'),
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
write_csv(grid, 'data/quant150k_grid.csv')
write_csv(val_df, 'data/quant150k_grid_val_mse.csv')

