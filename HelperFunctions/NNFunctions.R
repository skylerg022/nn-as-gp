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
## NOTE: Created model will work for continuous or binary response, but not categorical response.
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
## NOTE: Created model will work for continuous or binary response, but not categorical response.
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
# - x_train: nxp matrix of training observation predictors
# - y_train: nx1 matrix of response variable
# - x_val: mxp matrix of validation-split observation predictors
# - y_val: mx1 matrix of validation-split response variables
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
                     y_val = NULL, modeltype = 'custom', 
                     loss = loss_mean_squared_error(), test = 'part_train') {
  
  # Check for erroneous inputs
  if ( !(modeltype %in% c('custom', 'lee2018')) ) {
    err_message <- paste0('Model type not recognized. This determines how ',
                          'parameter \'pars\' will be used and what neural network ',
                          'framework will be used. Please use "custom" or ',
                          '"lee2018" for parameter \'modeltype.\'')
    stop(err_message)
  }
  # if (class(pars) != 'numeric') {
  #   err_message <- paste0('Input variable \'pars\' should be a numeric vector.')
  #   stop(err_message)
  # }
  modelpar_right <- ( (modeltype == 'custom' & length(pars) == 7) |
                      (modeltype == 'lee2018' & length(pars) == 9) )
  if (!modelpar_right) {
    err_message <- paste0('Input \'pars\' vector is not of correct length ',
                          'for specified \'modeltype\'.')
    stop(err_message)
  }
  if (nrow(x_train) != nrow(y_train)) {
    err_message <- paste0('\'x_train\' and \'y_train\' have a different number of rows.')
    stop(err_message)
  }
  if ( !(test %in% c('grid', 'part_train', 'all_train')) ) {
    err_message <- paste0('Model fitting test not recognized. Please assign test ',
                          'as "grid", "part_train", or "all_train".')
    stop(err_message)
  }

  
  # Build model according to modeltype
  if (modeltype == 'custom') {
    learning_rate <- 0.001
    epochs <- pars[[3]]
    batch_size <- pars[[4]]
    decay_rate <- pars[[5]]
    model_num <- pars[[7]]
    makeModFun <- makeModel
  } else if (modeltype == 'lee2018') {
    learning_rate <- pars[[3]]
    epochs <- pars[[5]]
    batch_size <- pars[[6]]
    decay_rate <- 0
    model_num <- pars[[9]]
    makeModFun <- makeModelLee2018
  }
  
  input_length <- ncol(x_train)
  
  model <- makeModFun(pars, input_length)
  model %>%
    compile(loss = loss, #loss = 'mse'
            optimizer = optimizer_adam(learning_rate = learning_rate,
                                       decay = decay_rate))
  
  if(test == 'grid') { # Evaluate performance on val set; no print to console
    t1 <- proc.time()
    history <- model %>% 
      fit(x_train, y_train, 
          epochs = epochs, batch_size = batch_size, 
          validation_data = list(x_val, y_val),
          view_metrics = FALSE,
          verbose = 0,
          callbacks = list(callback_early_stopping(monitor = 'val_loss',
                                                   patience = 5, 
                                                   restore_best_weights = TRUE)))
    t2 <- proc.time()
    t <- (t2 - t1)/60 # minutes
    
    print(sprintf('Model %.0f trained in %.1f minutes.', model_num, t[3]))
    result <- list(pars = pars,
                   val_loss = history$metrics$val_loss)
  } else if (test == 'part_train') { # Evaluate performance on val set
    history <- model %>% 
      fit(x_train, y_train, 
          epochs = epochs, batch_size = batch_size, 
          validation_data = list(x_val, y_val))
    
    try({beepr::beep()}, silent = TRUE) # Don't throw error if package not installed
    result <- model
  } else if (test == 'all_train') { # Fully train data
    model %>% 
      fit(x_train, y_train, 
          epochs = epochs, batch_size = batch_size)
    
    try({beepr::beep()}, silent = TRUE) # Don't throw error if package not installed
    result <- model
  }
  
  return(result)
}

