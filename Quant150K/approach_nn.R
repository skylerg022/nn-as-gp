library(tidyverse)
library(keras)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}
source('../HelperFunctions/MakeNNModel.R')
source('../HelperFunctions/Defaults.R')

# Read in data
load('data/SimulatedTempsSplit.RData')

fit_network <- function(type, model_pars, trainval_only = 0) {
  data_train <- rbind(cbind(x_train, y_train),
                      cbind(x_val, y_val))
  data_test <- cbind(x_test, y_test)
  
  # Transform data if needed
  if (type == 'nn') {
    # x_train and x_val are ready for scaling
  } else if (type == 'nn_trans') {
    x_train <- cbind(x_train, x_train^2)
    x_val <- cbind(x_val, x_val^2)
    x_test <- cbind(x_test, x_test^2)
  } else {
    err_message <- paste0('Grid search not recognized. Please assign type ',
                          'as "nn" or "nn_trans".')
    stop(err_message)
  }
  
  # Neural Network --------------------------------------------------------
  
  if( trainval_only == 1) {
    # Train on only training data and validate
    
    n_train <- nrow(x_train)
    # Center and scale train and val using training data only
    x_scaled <- predictorsScaled(x_train, x_val)
    x_train <- x_scaled[1:n_train,]
    x_val <- x_scaled[-c(1:n_train),]
    rm(x_scaled)
    
    model <- fitModel(model_pars, x_train, y_train, 
                      x_val, y_val, test = 'part_train')
    
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
    
    n <- nrow(data_train)
    x_scaled <- predictorsScaled(rbind(x_train, x_val), x_test, val_split = 0)
    x_train <- x_scaled[1:n,]
    x_test <- x_scaled[-c(1:n),]
    rm(x_scaled)
    
    y_train <- rbind(y_train, y_val)
    
    model <- fitModel(model_pars, x_train, y_train, test = 'all_train')
    
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
    yhat <- model %>%
      predict(x_test)
    rmse_test <- sqrt(mean( (y_test - yhat)^2 )) %>%
      round(2)
    
    # Evaluate train RMSE
    yhat <- model %>%
      predict(x_train)
    rmse_train <- sqrt(mean( (y_train - yhat)^2 )) %>%
      round(2)
    
    
    # Save ggplots
    cat('Saving prediction plots in ', getwd(), '/pics/\n', sep = '')
    filename <- paste0('pics/', type, '_rmsetrain', rmse_train,
                       '_test', rmse_test, '_showtest.png') %>%
      str_replace_all('(?<=[0-9])\\.(?=[0-9])', '_')
    ggsave(filename,
           plot = p_test,
           width = pic_width,
           height = pic_height,
           units = pic_units,
           bg = 'white')
    
    filename <- paste0('pics/', type, '_rmsetrain', rmse_train,
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


# One of the top five performers in sim_satellite nn grid search (1/15/2022)
model_pars <- c(n_layers = 16, layer_width = 2^6,
                epochs = 50, batch_size = 2^6,
                decay_rate = 0, dropout_rate = 0, 
                model_num = 25)

fit_nn <- fit_network(type = 'nn', model_pars = model_pars, trainval_only = 0)
fit_nn_trans <- fit_network(type = 'nn_trans', model_pars = model_pars, trainval_only = 0)
