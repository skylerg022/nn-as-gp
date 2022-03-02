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




evalNetwork <- function(x_train, y_train, x_val, y_val,
                        model_pars, x_test = NULL, y_test = NULL,
                        type = 'nn', pars_type = 'custom', 
                        sqrt_n_knots = NULL, plot = FALSE) {
  myenv <- environment()
  have_test_x <- !is.null(x_test)
  have_test_y <- !is.null(y_test)
  
  # Pre-transformed data for visualizations
  data_train <- rbind(cbind(x_train, y_train),
                      cbind(x_val, y_val))
  if (have_test_x) {
    if (have_test_y) {
      data_test <- cbind(x_test, y_test)
    } else {
      data_test <- x_test
    }
  }
  
  # Data Transformations (if needed) -----------------------------------------
  
  if (type == 'nn') {
    # x_train and x_val are ready for scaling
  } else if (type == 'nn_trans') {
    x_train <- cbind(x_train, x_train^2)
    x_val <- cbind(x_val, x_val^2)
    if (have_test_x == TRUE) {
      x_test <- cbind(x_test, x_test^2)
    }
  } else if (type == 'basis') {
    if (have_test_x == TRUE) {
      multiResBases(x_train = rbind(x_train, x_val),
                    x_withheld = x_test,
                    sqrt_n_knots = sqrt_n_knots,
                    thresh_type = 'colsum',
                    thresh = 30,
                    thresh_max = 0.75,
                    test = TRUE) %>%
        list2env(envir = myenv)
      # x_train <- x_bases$x_train
      # x_test <- x_bases$x_test
      # rm(x_bases)
    } else {
      multiResBases(x_train = x_train,
                    x_withheld = x_val,
                    sqrt_n_knots = sqrt_n_knots,
                    thresh_type = 'colsum',
                    thresh = 30,
                    thresh_max = 0.75,
                    test = FALSE) %>%
        list2env(envir = myenv)
      # x_train <- x_bases$x_train
      # x_val <- x_bases$x_test
      # rm(x_bases)
    }
  } else {
    err_message <- paste0('Grid search not recognized. Please assign type ',
                          'as "nn" or "nn_trans".')
    stop(err_message)
  }
  
  # Neural Network --------------------------------------------------------
  
  if( have_test_x == FALSE) {
    # # Train on only training data and validate
    # 
    # if (type != 'basis') {
    #   n_train <- nrow(x_train)
    #   # Center and scale train and val using training data only
    #   predictorsScaled(x_train, x_val, test = FALSE) %>%
    #     list2env(envir = myenv)
    #   # x_scaled <- predictorsScaled(x_train, x_val)
    #   # x_train <- x_scaled[1:n_train,]
    #   # x_val <- x_scaled[-c(1:n_train),]
    #   # rm(x_scaled)
    # }
    # 
    # model <- fitModel(model_pars, x_train, y_train, 
    #                   x_val, y_val, test = 'part_train',
    #                   modeltype = pars_type)
    # 
    # # Predictions after fitting 80% of training set
    # Predicted <- model %>%
    #   predict(rbind(x_train, x_val))
    # 
    # data_pred <- cbind(data_train, Predicted) %>%
    #   # clip predictions to max/min of observed data
    #   # mutate(Predicted = ifelse(Predicted > max(TrueTemp), max(TrueTemp), Predicted),
    #   #        Predicted = ifelse(Predicted < min(TrueTemp), min(TrueTemp), Predicted)) %>%
    #   pivot_longer(cols = c(TrueTemp, Predicted),
    #                names_to = 'type',
    #                values_to = 'z')
    # 
    # simple_train <- data_pred %>%
    #   mutate(across(c(x,y), round, digits = 2)) %>%
    #   group_by(x, y, type) %>%
    #   summarize(Temp = mean(z))
    # 
    # p <- simple_train %>%
    #   ggplot(aes(x, y, fill = Temp)) +
    #   geom_raster() +
    #   facet_wrap(~type) +
    #   scale_fill_continuous(type = 'viridis') +
    #   theme_minimal()
    # 
    # print(p)
    # 
    # val_rmse <- sqrt(mean( (y_val - Predicted[-c(1:n_train),])^2 ))
    # 
    # results <- list(model = model,
    #                 val_loss = val_rmse)
    cat('Training and visualizations only on val performance not updated. Update needed here.')
    results <- list(model = NULL,
                    val_loss = NULL)
  } else {
    
    # Test Predictions --------------------------------------------------------
    
    if (type != 'basis') {
      n <- nrow(data_train)
      predictorsScaled(rbind(x_train, x_val), x_test, test = TRUE) %>%
        list2env(envir = myenv)
      rm(x_val)
      # x_scaled <- predictorsScaled(rbind(x_train, x_val), x_test, val_split = 0)
      # x_train <- x_scaled[1:n,]
      # x_test <- x_scaled[-c(1:n),]
      # rm(x_scaled)
    }
    
    y_train <- rbind(y_train, y_val)
    
    model <- fitModel(model_pars, x_train, y_train, 
                      test = 'all_train',
                      modeltype = pars_type)
    
    if (have_test_y == TRUE) {
      # Predictions (including test set) after fitting 100% of training set
      Predicted <- model %>%
        predict(rbind(x_train, x_test))
      
      data_pred <- cbind(rbind(data_train, 
                               data_test), 
                         Predicted) %>%
        pivot_longer(cols = c(TrueTemp, Predicted),
                     names_to = 'type',
                     values_to = 'z')
      
      simple_train <- data_pred %>%
        mutate(across(c(x,y), round, digits = 2)) %>%
        group_by(x, y, type) %>%
        summarize(z = mean(z))
      
      # Just test performance
      p_test <- simple_train %>%
        ggplot(aes(x, y, fill = z)) +
        geom_raster() +
        geom_raster(data = data_train,
                    mapping = aes(x, y, fill = min(simple_train$z))) +
        facet_wrap(~type) +
        scale_fill_continuous(type = 'viridis') +
        theme_minimal() +
        labs(fill = 'Temp')
      
      # All data prediction
      p_all <- simple_train %>%
        ggplot(aes(x, y, fill = z)) +
        geom_raster() +
        facet_wrap(~type) +
        scale_fill_continuous(type = 'viridis') +
        theme_minimal() +
        labs(fill = 'Temp')
      
      # Evaluate test RMSE
      yhat_test <- model %>%
        predict(x_test)
      rmse_test <- sqrt(mean( (y_test - yhat_test)^2 )) %>%
        round(2)
      
      # Evaluate train RMSE
      yhat <- model %>%
        predict(x_train)
      rmse_train <- sqrt(mean( (y_train - yhat)^2 )) %>%
        round(2)
      
      
      # Save ggplots
      cat('Saving prediction plots in ', getwd(), '/pics/\n', sep = '')
      filename <- paste0('pics/', type, '_', paste(sqrt_n_knots, collapse = '_'),
                         '_', pars_type, '_rmsetrain', rmse_train,
                         '_test', rmse_test, '_showtest.png') %>%
        str_replace_all('(?<=[0-9])\\.(?=[0-9])', '_')
      ggsave(filename,
             plot = p_test,
             width = pic_width,
             height = pic_height,
             units = pic_units,
             bg = 'white')
      
      filename <- paste0('pics/', type, '_', paste(sqrt_n_knots, collapse = '_'),
                         '_', pars_type, '_rmsetrain', rmse_train,
                         '_test', rmse_test, '_showall.png') %>%
        str_replace_all('(?<=[0-9])\\.(?=[0-9])', '_')
      ggsave(filename,
             plot = p_all,
             width = pic_width,
             height = pic_height,
             units = pic_units,
             bg = 'white')
    }
    
    yhat_test <- model %>%
      predict(x_test)
    
    results <- list(model = model,
                    val_loss = NULL,
                    yhat_test = yhat_test)
  # } else {
  #   yhat_test <- model %>%
  #     predict(x_test)
  #   results <- list(model = model,
  #                   val_loss = NULL,
  #                   yhat_test = yhat_test)
  }
  return(results)
}
