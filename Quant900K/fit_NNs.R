library(tidyverse)
library(keras)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../HelperFunctions/Preprocess.R')
source('../HelperFunctions/NNFunctions.R')
source('../HelperFunctions/Defaults.R')

# Read in data
load('data/dataset2_split.RData')

fit_network <- function(type, model_pars, trainval_only = FALSE, pars_type = 'custom', sqrt_n_knots = NULL) {
  data_train <- rbind(cbind(x_train, y_train),
                      cbind(x_val, y_val))
  data_test <- x_test
  # data_test <- cbind(x_test, y_test)

  # Data Transformations (if needed) -----------------------------------------

  if (type == 'nn') {
    # x_train and x_val are ready for scaling
  } else if (type == 'nn_trans') {
    x_train <- cbind(x_train, x_train^2)
    x_val <- cbind(x_val, x_val^2)
    x_test <- cbind(x_test, x_test^2)
  } else if (type == 'basis') {
    if (trainval_only == TRUE) {
      multiResBases(x_train = x_train,
                    x_withheld = x_val,
                    sqrt_n_knots = sqrt_n_knots,
                    thresh_type = 'colsum',
                    thresh = 30,
                    thresh_max = 0.75,
                    test = FALSE) %>%
        list2env(envir = parent.frame())
      # x_train <- x_bases$x_train
      # x_val <- x_bases$x_test
      # rm(x_bases)
    } else {
      multiResBases(x_train = x_train,
                    x_withheld = x_test,
                    sqrt_n_knots = sqrt_n_knots,
                    thresh_type = 'colsum',
                    thresh = 30,
                    thresh_max = 0.75,
                    test = TRUE) %>%
        list2env(envir = parent.frame())
      # x_train <- x_bases$x_train
      # x_test <- x_bases$x_test
      # rm(x_bases)
    }
  } else {
    err_message <- paste0('Grid search not recognized. Please assign type ',
                          'as "nn" or "nn_trans".')
    stop(err_message)
  }
  
  # Neural Network --------------------------------------------------------
  
  if( trainval_only == TRUE) {
    # Train on only training data and validate
    
    if (type != 'basis') {
      n_train <- nrow(x_train)
      # Center and scale train and val using training data only
      predictorsScaled(x_train, x_val, test = FALSE) %>%
        list2env(envir = parent.frame())
      # x_scaled <- predictorsScaled(x_train, x_val)
      # x_train <- x_scaled[1:n_train,]
      # x_val <- x_scaled[-c(1:n_train),]
      # rm(x_scaled)
    }
    
    model <- fitModel(model_pars, x_train, y_train, 
                      x_val, y_val, test = 'part_train',
                      modeltype = pars_type)
    
    # Predictions after fitting 80% of training set
    Predicted <- model %>%
      predict(rbind(x_train, x_val))
    
    data_pred <- cbind(data_train, Predicted) %>%
      # clip predictions to max/min of observed data
      # mutate(Predicted = ifelse(Predicted > max(TrueTemp), max(TrueTemp), Predicted),
      #        Predicted = ifelse(Predicted < min(TrueTemp), min(TrueTemp), Predicted)) %>%
      pivot_longer(cols = c(TrueTemp, Predicted),
                   names_to = 'type',
                   values_to = 'z')
    
    simple_train <- data_pred %>%
      mutate(across(c(x,y), round, digits = 2)) %>%
      group_by(x, y, type) %>%
      summarize(Temp = mean(z))
    
    p <- simple_train %>%
      ggplot(aes(x, y, fill = Temp)) +
      geom_raster() +
      facet_wrap(~type) +
      scale_fill_continuous(type = 'viridis') +
      theme_minimal()
    
    print(p)
    
    val_rmse <- sqrt(mean( (y_val - Predicted[-c(1:n_train),])^2 ))
    
    results <- list(model = model,
                    val_loss = val_rmse)
  } else {
    
    # Test Predictions --------------------------------------------------------
    
    if (type != 'basis') {
      n <- nrow(data_train)
      predictorsScaled(rbind(x_train, x_val), x_test, test = TRUE) %>%
        list2env(envir = parent.frame())
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
    
    results <- list(model = model,
                    val_loss = NULL)
  }
  return(results)
}

# Model Fitting and Picture Generation ------------------------------------

# Lat-Long NN
model_pars_lee <- c(n_layers = 16, layer_width = 2^7, 
                    learning_rate = 0.0001533951,
                    weight_decay = 5.617919e-07, 
                    epochs = 30, batch_size = 2^4, 
                    sigma_w = 2.4846275, sigma_b = 0.3760813, 
                    model_num = 2487)
model_pars <- c(n_layers = 16, layer_width = 2^7,
                epochs = 90, batch_size = 2^6,
                decay_rate = 0, dropout_rate = 0,
                model_num = 45)

## Lee2018
fit <- fit_network(type = 'nn', model_pars = model_pars_lee,
                   trainval_only = FALSE, pars_type = 'lee2018')
## Custom
fit <- fit_network(type = 'nn', model_pars = model_pars, 
                   trainval_only = FALSE)


# Lat-Long Transformed NN
model_pars_lee <- c(n_layers = 2, layer_width = 2^7, 
                    learning_rate = 0.0065471031,
                    weight_decay = 4.375561e-08, 
                    epochs = 25, batch_size = 2^7, 
                    sigma_w = 2.2478299, sigma_b = 0.86696268, 
                    model_num = 3070)
model_pars <- c(n_layers = 16, layer_width = 2^7,
                epochs = 90, batch_size = 2^6,
                decay_rate = 0, dropout_rate = 0,
                model_num = 40)

## Lee2018
fit <- fit_network(type = 'nn_trans', model_pars = model_pars_lee,
                   trainval_only = FALSE, pars_type = 'lee2018')
## Custom
fit <- fit_network(type = 'nn_trans', model_pars = model_pars, 
                   trainval_only = FALSE)


# Basis 4by4
model_pars_lee <- c(n_layers = 10, layer_width = 2^8, 
                    learning_rate = 0.0054893277,
                    weight_decay = 1.442865e-07, 
                    epochs = 15, batch_size = 2^4, 
                    sigma_w = 1.258086, sigma_b = 1.302420401, 
                    model_num = 1249)
model_pars <- c(n_layers = 8, layer_width = 2^7,
                epochs = 40, batch_size = 2^6,
                decay_rate = 0, dropout_rate = 0.1,
                model_num = 67)

## Lee2018
fit <- fit_network(type = 'basis', model_pars = model_pars_lee,
                   trainval_only = FALSE, pars_type = 'lee2018',
                   sqrt_n_knots = c(4))
## Custom
fit <- fit_network(type = 'basis', model_pars = model_pars, 
                   trainval_only = FALSE, sqrt_n_knots = c(4))


# Basis 4by4&20by20
model_pars_lee <- c(n_layers = 1, layer_width = 2^7, 
                    learning_rate = 0.0029829550,
                    weight_decay = 3.655114e-05, 
                    epochs = 35, batch_size = 2^5, 
                    sigma_w = 0.8050873, sigma_b = 1.2809608, 
                    model_num = 2983)
model_pars <- c(n_layers = 8, layer_width = 2^7,
                epochs = 20, batch_size = 2^6,
                decay_rate = 3.828484e-05, dropout_rate = 0.1,
                model_num = 70)

## Lee2018
fit <- fit_network(type = 'basis', model_pars = model_pars_lee,
                   trainval_only = FALSE, pars_type = 'lee2018',
                   sqrt_n_knots = c(4, 20))
## Custom
fit <- fit_network(type = 'basis', model_pars = model_pars, 
                   trainval_only = FALSE, sqrt_n_knots = c(4, 20))

