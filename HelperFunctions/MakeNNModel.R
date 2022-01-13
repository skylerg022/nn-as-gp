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
    layer_dense(units = layer_width, input_length = c(input_length), activation = 'relu') %>%
    layer_dropout(rate = dropout_rate) %>%
    addLayers(n_layers - 1) %>%
    layer_dense(units = 1)
  
  return(model)
}

# fitModel
# Fit model with structural and learning parameters specified
#  as well as training data
# Input:
# - pars: a numeric vector of length 7 giving model parameters in 
#   the following order:
#   - n_layers, layer_width, epochs, batch_size, decay_rate,
#     dropout_rate, model_num
# - x_train: training
# - test: a string value of 'part_train', 'all_train', or 'grid'
# Output:
# - If test is 'part_train' or 'all_train', returns a customized neural
#   network model object
# - If test is 'grid', returns a list of model parameters and validation
#   loss record across epochs
fitModel <- function(pars, x_train, y_train, test = 'part_train') {
  # Constants
  VAL_SPLIT <- 0.2
  LEARNING_RATE <- 0.001
  
  epochs <- pars[[3]]
  batch_size <- pars[[4]]
  decay_rate <- pars[[5]]
  model_num <- pars[[7]]
  
  input_length <- ncol(x_train)
  
  model <- makeModel(pars, input_length)
  model %>%
    compile(loss = 'mse',
            optimizer = optimizer_adam(learning_rate = LEARNING_RATE,
                                       decay = decay_rate))
  
  if(test == 'grid') { # Evaluate performance on val set; no print to console
    history <- model %>% 
      fit(x_train, y_train, 
          epochs = epochs, batch_size = batch_size, 
          validation_split = VAL_SPLIT,
          view_metrics = FALSE,
          verbose = 0)
    print(sprintf('Model %.0f Trained', model_num))
    list(pars = pars,
         val_loss = history$metrics$val_loss)
  } else if (test == 'part_train') { # Evaluate performance on val set
    history <- model %>% 
      fit(x_train, y_train, 
          epochs = epochs, batch_size = batch_size, 
          validation_split = VAL_SPLIT)
    beepr::beep()
    model
  } else if (test == 'all_train') { # Fully train data
    model %>% 
      fit(x_train, y_train, 
          epochs = epochs, 
          batch_size = batch_size)
    beepr::beep()
    model
  } else {
    err_message <- paste0('Model fitting test not recognized. Please assign test ',
                          'as "grid", "part_train", or "all_train".')
    stop(err_message)
  }
}
