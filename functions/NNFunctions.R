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
makeModel <- function(pars, input_length, binary_data = FALSE) {
  
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
    addLayers(n_layers - 1)
  
  if(binary_data == TRUE) {
    model <- model %>%
      layer_dense(units = 1, activation = 'sigmoid')
  }
  else {
    model <- model %>%
      layer_dense(units = 1)
  }
  
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
makeModelLee2018 <- function(pars, input_length, binary_data = FALSE) {
  
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
    addLayers(n_layers - 1)
  
  if(binary_data == TRUE) {
    model <- model %>%
      layer_dense(units = 1, activation = 'sigmoid')
  }
  else {
    model <- model %>%
      layer_dense(units = 1)
  }
  
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
                     binary_data = FALSE,
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
  
  model <- makeModFun(pars, input_length, binary_data = binary_data)
  model %>%
    compile(loss = loss,
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


# evalNetwork
# Evaluate network performance given model parameters and data
# Input:
# - pars: a numeric vector of varying length depending on the 'modeltype'
#   parameter. See modeltype description below.
# - x_train: nxp matrix of training observation predictors
# - y_train: nx1 matrix of response variable
# - x_val: mxp matrix of validation-split observation predictors
# - y_val: mx1 matrix of validation-split response variables
# Output:
# - Saves two prediction plots, one with both train and test, another with only test
# - If y_test is not null, saves two error plots, one with train and test, another with only test
evalNetwork <- function(x_train, y_train, x_val, y_val,
                        model_pars, x_test, y_test = NULL,
                        binary_data = FALSE,
                        type = 'nn', pars_type = 'custom', 
                        sqrt_n_knots = NULL) {
  myenv <- environment()
  have_test_y <- !is.null(y_test)
  
  # New loss if data is binary
  if(binary_data == TRUE) {
    loss <- loss_binary_crossentropy()
  } else {
    loss <- loss_mean_squared_error()
  }
  
  # raw data for visualizations
  data_train <- rbind(cbind(x_train, Truth = c(y_train)),
                      cbind(x_val, Truth = c(y_val))) %>%
    as.data.frame()
  y_train <- rbind(y_train, y_val)
  rm(y_val)
  
  if(!is.null(y_test)) {
    data_test <- cbind(x_test, Truth = c(y_test)) %>%
      as.data.frame()
  } else {
    data_test <- cbind(x_test, Truth = NA) %>%
      as.data.frame()
  }
  
  
  
  # Location Transformations (if needed) -----------------------------------------
  
  if (type == 'nn') {
    # Standardize when not using a radial basis function expansion
    n <- nrow(data_train)
    predictorsScaled(rbind(x_train, x_val), x_test, test = TRUE) %>%
      list2env(envir = myenv)
    # x_train and x_val are ready for scaling
  } else if (type == 'nn_trans') {
    # Standardize when not using a radial basis function expansion
    n <- nrow(data_train)
    predictorsScaled(rbind(x_train, x_val), x_test, test = TRUE) %>%
      list2env(envir = myenv)
    x_train <- cbind(x_train, x_train^2, sin(x_train), cos(x_train))
    x_val <- cbind(x_val, x_val^2, sin(x_val), cos(x_val))
    x_test <- cbind(x_test, x_test^2, sin(x_test), cos(x_test))
  } else if (type == 'basis') {
    multiResBases(x_train = rbind(x_train, x_val),
                  x_withheld = x_test,
                  sqrt_n_knots = sqrt_n_knots,
                  local_n = 30,
                  closest_minval = 0.75,
                  test = TRUE) %>%
      list2env(envir = myenv)
  } else {
    err_message <- paste0('Grid search not recognized. Please assign type ',
                          'as "nn", "nn_trans", or "basis".')
    stop(err_message)
  }
  rm(x_val)
  
  # Neural Network --------------------------------------------------------
  
  # Fit name for saving plots
  if(!is.null(sqrt_n_knots)) {
    fit_name <- paste(type, paste0(sqrt_n_knots, collapse = '_'), 
                      pars_type, sep = '_')
  } else {
    fit_name <- paste(type, pars_type, sep = '_')
  }
  
  model <- fitModel(model_pars, x_train, y_train,
                    test = 'all_train',
                    modeltype = pars_type,
                    binary_data = binary_data,
                    loss = loss)
  
  n_train <- nrow(x_train)
  yhat_train <- predict(model, x_train)
  yhat_test <- predict(model, x_test)
  
  # Save plots and prepare statistics
  if (binary_data == TRUE) { # Assuming EVEN spacing across observation locations
    data_pred <- cbind(rbind(data_train, data_test), 
                       Predicted = ifelse(c(yhat_train, yhat_test) > 0.5, 1, 0))
    
    # Save training metrics to return from function
    BinaryMetrics <- function(predicted, truth) {
      # Assessing train metrics
      levels <- c('1', '0')
      cm <- caret::confusionMatrix(factor(predicted, levels = levels), 
                                   factor(truth, levels = levels))
      acc <- unname(cm$overall['Accuracy'])
      mets <- unname(cm$byClass[c('Pos Pred Value', 'Sensitivity', 'F1')])
      # Calculate train metrics
      mymets <- list(rmse = NA,
                     accuracy = acc,
                     precision = mets[1],
                     recall = mets[2],
                     f1 = mets[3])
      
      return(mymets)
    }
    train_metrics <- BinaryMetrics(data_pred$Predicted[1:n_train],
                                   data_pred$Truth[1:n_train])
    test_metrics <- NA
      
    # Plots
    # Train & test prediction plot
    PredPlot <- function(data_pred) {
      p <- data_pred %>%
        ggplot(aes(x, y, fill = as.factor(Predicted))) +
        geom_raster() +
        scale_fill_manual(values = colors_binary) +
        labs(fill = 'Predicted') +
        theme(panel.background = element_rect(fill = 'gray30'),
              panel.grid = element_blank())
      return(p)
    }
    
    p <- PredPlot(data_pred)
    
    myggsave(paste0('pics/final/', fit_name, '_all.pdf'), plot = p)
    myggsave(paste0('pics/final/', fit_name, '_all.png'), plot = p)
    
    # Make error plot for all observations
    if (have_test_y == TRUE) {
      # Retrieve and save test metrics for returning
      test_metrics <- BinaryMetrics(data_pred$Predicted[-c(1:n_train)],
                                    data_pred$Truth[-c(1:n_train)])
      
      data_raster <- data_pred %>%
        mutate(error = Predicted - Truth,
               error = case_when(error == -1 ~ 'Type II',
                                 error == 1 ~ 'Type I',
                                 TRUE       ~ 'Correct') %>%
                 factor(levels = c('Correct', 'Type I', 'Type II')))
      
      ErrorPlot <- function(data_raster) {
        mycols <- qualitative_hcl(3, h = 20, c = 80, l = 60)
        p <- data_raster %>%
          ggplot(aes(x, y,fill = error)) +
          geom_raster() +
          scale_fill_manual(values = c(mycols[2], mycols[1], mycols[3])) +
          labs(fill = 'Error') +
          theme(panel.background = element_rect(fill = 'gray30'),
                panel.grid = element_blank())
        
        return(p)
      }
      
      p <- ErrorPlot(data_raster)
      
      myggsave(paste0('pics/final/', fit_name, '_error_all.pdf'), plot = p)
      myggsave(paste0('pics/final/', fit_name, '_error_all.png'), plot = p)
    }
    
    # Test only prediction plot
    data_pred <- cbind(data_test, 
                       Predicted = ifelse(c(yhat_test) > 0.5, 1, 0))
    
    p <- PredPlot(data_pred)
    
    myggsave(paste0('pics/final/', fit_name, '_test.pdf'), plot = p)
    myggsave(paste0('pics/final/', fit_name, '_test.png'), plot = p)
    
    # Make error plot for just test observations
    if (have_test_y == TRUE) {
      data_raster <- data_pred %>%
        mutate(error = Predicted - Truth,
               error = case_when(error == -1 ~ 'Type II',
                                 error == 1 ~ 'Type I',
                                 TRUE       ~ 'Correct') %>%
                 factor(levels = c('Correct', 'Type I', 'Type II')))
      
      p <- ErrorPlot(data_raster)
      
      myggsave(paste0('pics/final/', fit_name, '_error_test.pdf'), plot = p)
      myggsave(paste0('pics/final/', fit_name, '_error_test.png'), plot = p)
    }
    
    # Return train and (if available) test metrics
    metrics <- list(train = train_metrics,
                    test = test_metrics)
    
  } else { # Assuming UNEVEN spacing across observation locations
    data_pred <- cbind(rbind(data_train, data_test), 
                       Predicted = c(yhat_train, yhat_test))
    
    # Save training metrics to return from function
    RMSE <- function(predicted, truth) {
      return(sqrt(mean( (predicted - truth)^2 )))
    }
    train_metrics <- list(rmse = RMSE(data_pred$Predicted[1:n_train],
                                      data_pred$Truth[1:n_train]),
                          accuracy = NA,
                          precision = NA,
                          recall = NA,
                          f1 = NA)
    test_metrics <- NA
    
    # Plots
    # Train & test prediction plot
    data_raster <- data_pred %>%
      mutate(across(c(x,y), discretize, nbins = 300)) %>%
      group_by(x, y) %>%
      summarize(z = mean(Predicted))
    
    p <- data_raster %>%
      ggplot(aes(x, y, fill = z)) +
      geom_raster() +
      scale_fill_continuous(type = 'viridis',
                            limits = c(min(y_train, yhat_train, yhat_test),
                                       max(y_train, yhat_train, yhat_test))) +
      labs(fill = 'Predicted') +
      theme(panel.background = element_rect(fill = 'gray30'),
            panel.grid = element_blank())
    
    myggsave(paste0('pics/final/', fit_name, '_all.pdf'), plot = p)
    myggsave(paste0('pics/final/', fit_name, '_all.png'), plot = p)
    
    # Save test performance and error plot if possible
    if (have_test_y == TRUE) {
      # Produce test performance metrics
      test_metrics <- list(rmse = RMSE(data_pred$Predicted[-c(1:n_train)],
                                       data_pred$Truth[-c(1:n_train)]),
                           accuracy = NA,
                           precision = NA,
                           recall = NA,
                           f1 = NA)
      
      # Make error plot
      data_raster <- data_pred %>%
        mutate(across(c(x,y), discretize, nbins = 300),
               error = Predicted - Truth) %>%
        group_by(x, y) %>%
        summarize(error = mean(error))
      
      p <- data_raster %>%
        ggplot(aes(x, y, fill = error)) +
        geom_raster() +
        scale_fill_gradient2() +
        labs(fill = 'Error') +
        theme(panel.background = element_rect(fill = 'gray30'),
              panel.grid = element_blank())
      
      myggsave(paste0('pics/final/', fit_name, '_error_all.pdf'), plot = p)
      myggsave(paste0('pics/final/', fit_name, '_error_all.png'), plot = p)
    }
    
    # Test only prediction plot
    data_pred <- cbind(data_test, 
                       Predicted = c(yhat_test))
    
    data_raster <- data_pred %>%
      mutate(across(c(x,y), discretize, nbins = 300)) %>%
      group_by(x, y) %>%
      summarize(z = mean(Predicted))
    
    p <- data_raster %>%
      ggplot(aes(x, y, fill = z)) +
      geom_raster() +
      scale_fill_continuous(type = 'viridis',
                            limits = c(min(y_train, yhat_train, yhat_test),
                                       max(y_train, yhat_train, yhat_test))) +
      labs(fill = 'Predicted') +
      theme(panel.background = element_rect(fill = 'gray30'),
            panel.grid = element_blank())
    
    myggsave(paste0('pics/final/', fit_name, '_test.pdf'), plot = p)
    myggsave(paste0('pics/final/', fit_name, '_test.png'), plot = p)
    
    if (have_test_y == TRUE) {
      # Make error plot
      data_raster <- data_pred %>%
        mutate(across(c(x,y), discretize, nbins = 300),
               error = Predicted - Truth) %>%
        group_by(x, y) %>%
        summarize(error = mean(error))
      
      p <- data_raster %>%
        ggplot(aes(x, y, fill = error)) +
        geom_raster() +
        scale_fill_gradient2() +
        labs(fill = 'Error') +
        theme(panel.background = element_rect(fill = 'gray30'),
              panel.grid = element_blank())
      
      myggsave(paste0('pics/final/', fit_name, '_error_test.pdf'), plot = p)
      myggsave(paste0('pics/final/', fit_name, '_error_test.png'), plot = p)
    }
    
    metrics <- list(train = train_metrics,
                    test = test_metrics)
  }
  
  return(list(model = model,
              metrics = metrics))
}

