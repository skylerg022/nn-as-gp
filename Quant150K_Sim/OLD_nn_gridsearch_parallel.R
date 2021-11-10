library(tidyverse)
library(keras)
library(parallel)

# Read in Data ------------------------------------------------------------

# set working directory if necessary
tryCatch(setwd("C:/Users/skyle/Dropbox/2021 Fall/research/Quant150K"),
         error = function(cond) {
           message(paste0('Could not set directory. ',
                          'Assuming code is being run via Bash.'))
         })
# Read in data
load('data/AllSimulatedTemps.RData')
data_train <- all.sim.data %>%
  filter(!is.na(MaskTemp))
data_test <- all.sim.data %>%
  filter(is.na(MaskTemp))
rm(all.sim.data)


# Neural Network --------------------------------------------------------

x_train <- data_train[,1:2] %>%
  as.matrix()
y_train <- data_train[,3, drop = FALSE] %>%
  as.matrix()
x_test <- data_test[,1:2] %>%
  as.matrix()

VAL_SPLIT <- 0.2
TRAIN_SIZE <- (1 - VAL_SPLIT) * nrow(x_train)

grid <- expand.grid(n_layers = c(1, 2, 4, 8), 
                    layer_width = c(2^3, 2^6, 2^7),
                    epochs = c(30, 60, 200), 
                    batch_size = c(2^6, 2^7, 2^8),
                    decay_rate = c(0, 0.05),
                    square_feat = c(0, 1),
                    dropout_rate = c(0, 0.5)) %>%
  mutate(decay_rate = decay_rate / 
           (TRAIN_SIZE %/% batch_size))

# NORMAL TIME TO PROCESS ONE OF THE NN PARAMETER SETS
# time <- system.time({
#   set.seed(1812)
#   fitModel(grid[1,], x_train, y_train)
# })



# PARALLEL PROCESSING IMPLEMENTATION (DOESN'T IMPROVE SPEED; MAYBE NOT COMPLICATED ENOUGH?)
# numCores <- detectCores(logical = FALSE) # I have 4 physical cores, 8 logical cores
# 
# # TEST settings
# numCores <- 2
# # grid[1:3, 3] <- 1
# grid[1:3, 4] <- 128
# 
# time <- system.time({
#   cl <- makeCluster(numCores) # initialize cores
#   clusterEvalQ(cl, { # load library in cores
#     library(tidyverse)
#     library(keras)
#     library(parallel)
#   })
#   
#   clusterExport(cl, c('x_train', 'y_train',
#                       'makeModel', 'fitModel'))
#   
#   results <- parApply(cl, grid[1:2,], 1, 
#            function(x) fitModel(x, x_train, y_train))
#   
#   stopCluster(cl)
#   })
# 
# results

# time_notparallel <- system.time({
#   apply(grid[1:2,], 1, function(x) fitModel(x, x_train, y_train))
#   })

# TEST: only two of 144 parameter sets
grid_list <- split(grid[1:10,], 1:10)

n_cores <- detectCores()
time <- system.time({
  results <- mclapply(grid_list, function(x) fitModel(x, x_train, y_train),
                      mc.cores = n_cores)
})

nn_mse <- numeric(length(results))
for (i in 1:length(results)) nn_mse[i] <- tail(results[[i]]$val_loss, n=1)
best_par <- which.min(nn_mse)

results[[best_par]]$pars
tail(results[[best_par]]$val_loss, n=1)


fileConn <- file("log_gridfit.txt")
open(fileConn, open = 'a')
writeLines(c("Hello","World"), fileConn)
close(fileConn)

a[[1]]
class(a)
unlist(a, recursive = FALSE) %>%
  tidyr::tibble() %>%
  
  t() %>%
  tidyr::pivot_longer()
