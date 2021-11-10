library(tidyverse)
library(keras)
library(parallel)

# Model Fitting Functions -------------------------------------------------

# Make a model with varying # of hidden layers and width
# Input: lat and long coordinates
# Output: yhat on the real line
makeModel <- function(pars) {
  
  n_layers <- pars[[1]]
  layer_width <- pars[[2]]
  
  addLayers <- function(model, n_layers) {
    if (n_layers > 0) {
      addLayers(model, n_layers-1) %>%
        layer_dense(units = layer_width, activation = 'relu')
    } else model
  }
  
  # Set up compiler
  model <- keras_model_sequential() %>%
    layer_dense(units = layer_width, input_shape = c(2), activation = 'relu') %>%
    addLayers(n_layers - 1) %>%
    layer_dense(units = 1)
  model
}

# fitModel
# Fit model with structural and learning parameters specified
#  as well as training data
fitModel <- function(pars, x_train, y_train) {
  # Constants
  VAL_SPLIT <- 0.2
  LEARNING_RATE <- 0.001
  
  epochs <- pars[[3]]
  batch_size <- pars[[4]]
  decay_rate <- pars[[5]]
  
  model <- makeModel(pars)
  model %>%
    compile(loss = 'mse',
            optimizer = optimizer_adam(learning_rate = LEARNING_RATE,
                                       decay = decay_rate))
  # Evaluate performance on validation
  history <- model %>% 
    fit(x_train, y_train, 
        epochs = epochs, batch_size = batch_size, 
        validation_split = VAL_SPLIT,
        view_metrics = FALSE,
        verbose = 0)
  list(pars = pars,
       val_loss = history$metrics$val_loss)
}



# Read in Data ------------------------------------------------------------

dataset_num <- 1
data_train <- read_csv( paste0('data/dataset', dataset_num, '_training.csv') )
data_test <- read_csv( paste0('data/dataset', dataset_num, '_testing.csv') )


# Neural Network --------------------------------------------------------

x_train <- data_train[,1:2] %>%
  as.matrix()
y_train <- data_train[,3, drop = FALSE] %>%
  as.matrix()
x_test <- data_test %>%
  as.matrix()

VAL_SPLIT <- 0.2
TRAIN_SIZE <- (1 - VAL_SPLIT) * nrow(x_train)

grid <- expand.grid(n_layers = c(1, 2, 4, 8), 
                    layer_width = c(2^3, 2^6, 2^7),
                    epochs = c(30, 60, 200), 
                    batch_size = c(2^6, 2^7, 2^8),
                    decay_rate = c(0, 0.05),
                    square_feat = c(0, 1)) %>%
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
grid_list <- split(grid[1:2,], seq(nrow(grid[1:2,])))

time <- system.time({
  results <- mclapply(grid_list, function(x) fitModel(x, x_train, y_train))
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
