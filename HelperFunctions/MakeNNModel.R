## MakeNNModel.R contains code for fitting and predicting with
## model structures used in this analysis.

library(tidyverse)
library(keras)

# Model Fitting Functions -------------------------------------------------

# Make a model following model parameter specifications
# Input:
# - pars: a numeric vector of length 7 giving model parameters in 
#   the following order:
#   - n_layers, layer_width, epochs, batch_size, decay_rate,
#     dropout_rate, model_num
# Output: Customized neural network model object
makeModel <- function(pars, input_length) {
  
  n_layers <- pars[[1]]
  layer_width <- pars[[2]]
  dropout_rate <- pars[[6]]
  
  addLayers <- function(model, n_layers) {
    if (n_layers > 0) {
      model <- addLayers(model, n_layers-1) %>%
        layer_dense(units = layer_width, activation = 'relu')
      if (dropout_rate > 0) {
        model <- model %>%
          layer_dropout(rate = dropout_rate)
      }
    }
    model
  }
  
  # Set up compiler
  model <- keras_model_sequential() %>%
    layer_dense(units = layer_width, input_shape = c(input_length), activation = 'relu') %>%
    layer_dropout(rate = dropout_rate) %>%
    addLayers(n_layers - 1) %>%
    layer_dense(units = 1)
  
  return(model)
}

# Make a model following model parameter specifications
# Input:
# - pars: a numeric vector of length 9 giving model parameters in 
#   the following order:
#   - n_layers, layer_width, learning_rate, weight_decay, epochs, 
#     batch_size, sigma_w, sigma_b, model_num
# Output: Customized neural network model object
makeModelLee2018 <- function(pars, input_length) {
  
  n_layers <- pars[[1]]
  layer_width <- pars[[2]]
  learning_rate <- pars[[3]]
  weight_decay <- pars[[4]]
  epochs <- pars[[5]]
  batch_size <- pars[[6]]
  sigma_w <- pars[[7]]
  sigma_b <- pars[[8]]
  model_num <- pars[[9]]
  
  addLayers <- function(model, n_layers) {
    if (n_layers > 0) {
      model <- addLayers(model, n_layers-1) %>%
        layer_dense(units = layer_width, activation = 'relu',
                    kernel_initializer = initializer_random_normal(stddev = sigma_w/sqrt(layer_width)),
                    bias_initializer = initializer_random_normal(stddev = sigma_b),
                    kernel_regularizer = regularizer_l2(weight_decay),
                    bias_regularizer = regularizer_l2(weight_decay))
    }
    model
  }
  
  # Set up compiler
  model <- keras_model_sequential() %>%
    layer_dense(units = layer_width, input_shape = c(input_length), 
                activation = 'relu',
                kernel_initializer = initializer_random_normal(stddev = sigma_w/sqrt(layer_width)),
                bias_initializer = initializer_random_normal(stddev = sigma_b),
                kernel_regularizer = regularizer_l2(weight_decay),
                bias_regularizer = regularizer_l2(weight_decay)) %>%
    addLayers(n_layers - 1) %>%
    layer_dense(units = 1)
  
  return(model)
}


# fitModel
# Fit model with structural and learning parameters specified
#  as well as training data
# Input:
# - pars: a numeric vector of varying length depending on the 'modeltype'
#   parameter. See modeltype description below.
#   length 7 giving model parameters in 
#   the following order:
#   - 
# - x_train: training
# - modeltype: a string value of 'custom' or 'lee2018'. The type 'custom'
#   will expect a pars vector of length 7 for the following values
#   - n_layers, layer_width, epochs, batch_size, decay_rate,
#     dropout_rate, model_num
#   If modeltype is 'lee2018', the pars vector will include parameters
#   needed to create a neural network similar to the fully connected
#   neural nets in the Lee 2018 paper. The pars vector will then be of 
#   length 9 with the following elements:
#   - n_layers, layer_width, learning_rate, weight_decay, epochs, 
#     batch_size, sigma_w, sigma_b, model_num
# - test: a string value of 'part_train', 'all_train', or 'grid'
# Output:
# - If test is 'part_train' or 'all_train', returns a customized neural
#   network model object
# - If test is 'grid', returns a list of model parameters and validation
#   loss record across epochs
fitModel <- function(pars, x_train, y_train, x_val = NULL, 
                     y_val = NULL, modeltype = 'custom', test = 'part_train') {
  
  if (modeltype == 'custom') {
    learning_rate <- 0.001
    epochs <- pars[[3]]
    batch_size <- pars[[4]]
    decay_rate <- pars[[5]]
    model_num <- pars[[7]]
    makeModFun <- makeModel
  } else if (modeltype == 'lee2018') {
    # n_layers <- pars[[1]]
    learning_rate <- pars[[3]]
    epochs <- pars[[5]]
    batch_size <- pars[[6]]
    decay_rate <- 0
    model_num <- pars[[9]]
    makeModFun <- makeModelLee2018
  } else {
    err_message <- paste0('Model type not recognized. This determines how ',
                          'parameter \'pars\' will be used and what neural network ',
                          'framework will be used. Please use "custom" or ',
                          '"lee2018" for parameter \'modeltype.\'')
    stop(err_message)
  }
  
  input_length <- ncol(x_train)
  
  model <- makeModFun(pars, input_length)
  model %>%
    compile(loss = 'mse',
            optimizer = optimizer_adam(learning_rate = learning_rate,
                                       decay = decay_rate))
  
  if(test == 'grid') { # Evaluate performance on val set; no print to console
    history <- model %>% 
      fit(x_train, y_train, 
          epochs = epochs, batch_size = batch_size, 
          validation_data = list(x_val, y_val),
          view_metrics = FALSE,
          verbose = 0,
          callbacks = list(callback_early_stopping(monitor = 'val_loss',
                                                   patience = 5, 
                                                   restore_best_weights = TRUE)))
    print(sprintf('Model %.0f Trained', model_num))
    list(pars = pars,
         val_loss = history$metrics$val_loss)
  } else if (test == 'part_train') { # Evaluate performance on val set
    history <- model %>% 
      fit(x_train, y_train, 
          epochs = epochs, batch_size = batch_size, 
          validation_data = list(x_val, y_val))
    
    try({beepr::beep()}, silent = TRUE)
    model
  } else if (test == 'all_train') { # Fully train data
    model %>% 
      fit(x_train, y_train, 
          epochs = epochs, batch_size = batch_size)
    try({beepr::beep()}, silent = TRUE)
    model
  } else {
    err_message <- paste0('Model fitting test not recognized. Please assign test ',
                          'as "grid", "part_train", or "all_train".')
    stop(err_message)
  }
}



# Other functions ---------------------------------------------------------

# predictorStats
# Center and scale all observations by columns. To scale the data
#  using all observations, let x_test be NULL and set val_split to 0.
#  When x_test is not null, val_split is ignored.
# Inputs:
# - x_train: An n x p matrix of n observations and p numeric predictors.
# - x_test: An m x p matrix of m observations and p numeric predictors.
#     These observations will never be used to calculate column means and
#     standard deviations.
# - val_split: the proportion of training data to exclude (last 100*val_split 
#     percent of the observations) when calculating column means and standard
#     devations.
# Output:
# - An (n+m) x p matrix of all observations, centered and scaled using
#     just the training data.
predictorsScaled <- function(x_train, x_test = NULL, val_split = 0.2) {
  n <- nrow(x_train)
  if (is.null(x_test)) {
    n_train <- floor(nrow(x_train) * (1-val_split))
  } else {
    n_train <- n
  }
  mean_train <- x_train[1:n_train,] %>%
    colMeans()
  sd_train <- x_train[1:n_train,] %>%
    apply(2, sd)
  
  # Function to help with matrix math
  quickMatrix <- function(n, x) {
    matrix(x, ncol = length(x), nrow = n, byrow = TRUE)
  }
  
  # Scale predictors with appropriate statistics
  x_train <- ( (x_train - quickMatrix(n, mean_train)) /
                 quickMatrix(n, sd_train) ) %>%
    as.matrix()
  if (!is.null(x_test)) {
    x_test <- ( (x_test - quickMatrix(nrow(x_test), mean_train)) /
                  quickMatrix(nrow(x_test), sd_train) ) %>%
      as.matrix()
  }
  
  return(rbind(x_train, x_test))
}

