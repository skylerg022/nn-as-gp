
# Load necessary libraries and functions if not loaded already
library(tidyverse)
library(keras)
library(parallel)


# Setup Functions ---------------------------------------------------------

makeGridLee2018 <- function(n_layers = c(1, 3, 5, 10),
                            layer_width = c(2^7, 2^8, 2^9, 2^10, 2^11)) {
  temp <- expand.grid(n_layers = n_layers,
                      layer_width = layer_width)
  grid <- do.call("rbind", 
                  replicate(50, temp, simplify = FALSE))
  n_grid <- nrow(grid)
  
  grid <- grid %>%
    mutate(learning_rate = exp( runif(n_grid, log(10^(-4)), log(0.2)) ),
           weight_decay = exp( runif(n_grid, log(10^(-8)), log(1)) ),
           epochs = 100,
           batch_size = sample(c(2^4, 2^5, 2^6, 2^7, 2^8),
                               size = n_grid, replace = TRUE),
           sigma_w = runif(n_grid, 0.01, 2.5),
           sigma_b = runif(n_grid, 0, 1.5)) %>%
    # filter out observations with more than 800,000 parameters
    filter((layer_width^2)*(n_layers-1) <= 8e5) %>%
    arrange( desc((layer_width^2)*(n_layers-1)) ) %>%
    mutate(model_num = row_number())
  
  return(grid)
}


# gridsearch function for NN setups
gridsearch <- function(input_type = 'nn', modeltype = 'custom', 
                       n_cores = 20, sqrt_n_knots = NULL,
                       binary_data = FALSE,
                       loss = loss_mean_squared_error()) {
  myenv <- environment()
  
  # Input preprocessing
  if (input_type == 'nn') {
    # x_train and x_val are ready for scaling
  } else if (input_type == 'nn_trans') {
    x_train <- cbind(x_train, x_train^2)
    x_val <- cbind(x_val, x_val^2)
  } else if (input_type == 'basis') {
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
  } else {
    err_message <- paste0('Grid search not recognized. Please assign input_type ',
                          'as "nn" or "nn_trans".')
    stop(err_message)
  }
  
  # Neural Network
  
  n_train <- nrow(x_train)
   
  if (input_type != 'basis') {
    # Center and scale train and val using training data only
    predictorsScaled(x_train, x_val, test = FALSE) %>%
      list2env(envir = myenv)
    # x_scaled <- predictorsScaled(x_train, x_val)
    # x_train <- x_scaled[1:n_train,]
    # x_val <- x_scaled[-c(1:n_train),]
  }
  
  if (modeltype == 'custom') {
    grid <- expand.grid(n_layers = c(1, 2, 4, 8, 16), 
                        layer_width = c(2^3, 2^6, 2^7, 2^8, 2^9, 2^10, 2^11),
                        epochs = c(100), 
                        batch_size = c(2^4, 2^5, 2^6, 2^7, 2^8),
                        decay_rate = c(0, 0.01),
                        dropout_rate = c(0, 0.1)) %>%
      filter(layer_width^2 * (n_layers - 1) < 800000) %>%
      mutate(decay_rate = decay_rate / (n_train %/% batch_size),
             model_num = 1:n())
  } else if (modeltype == 'lee2018') {
    grid <- makeGridLee2018(n_layers = c(1, 2, 4, 8, 16),
                            layer_width = c(2^6, 2^7, 2^8, 2^9, 2^10, 2^11))
  } else {
    err_message <- paste0('Neural network modeltype not recognized. Please ', 
                          'assign modeltype to "custom" or "lee2018."')
    stop(err_message)
  }
  
  # Make grid into input class: list
  grid_list <- split(grid, 1:nrow(grid))
  
  # Use at most the number of cores available on server
  time <- system.time({
    results <- mclapply(grid_list, 
                        function(pars) fitModel(pars, x_train, y_train, 
                                                x_val, y_val, 
                                                modeltype = modeltype,
                                                binary_data = binary_data,
                                                test = 'grid', loss = loss),
                        mc.cores = n_cores, mc.silent = FALSE)
    # mc.cleanup = FALSE, mc.allow.recursive = FALSE)
  })
  
  # Convert results from list into dataframe
  val_df <- matrix(NA, nrow = 0, ncol = 3,
                   dimnames = list(NULL, c('model_num', 'epoch', 'val_loss')))
  for (i in 1:length(results)) {
    epochs <- length(results[[i]]$val_loss)
    new_rows <- data.frame(model_num = results[[i]]$pars[['model_num']],
                           epoch = 1:epochs,
                           val_loss = results[[i]]$val_loss)
    val_df <- rbind(val_df, new_rows)
  }
  
  # Write data to csv
  if (input_type == 'basis') {
    basis_knots <- sapply(sqrt_n_knots, function(x) { 
      num <- as.character(x)
      return(paste0(num, 'by', num))
      }) %>% paste(collapse = '_') %>%
      paste0('_')
  } else {
    basis_knots <- ''
  }
  write_csv(grid, paste0('data/grid_', input_type, '_', basis_knots, modeltype, '.csv'))
  write_csv(val_df, paste0('data/grid_', input_type, '_', basis_knots, modeltype, '_val_loss.csv'))
  
  return()
}



# Post-search EDA Functions ------------------------------------------------

# Create a raster for all combinations of xvar and yvar hyperparameters
customRaster <- function(data, xvar, yvar, fillvar,
                         xlab, ylab, fill_lab) {
  xvals <- pull(data, xvar)
  xvals_u <- sort(unique(xvals))
  yvals <- pull(data, yvar)
  yvals_u <- sort(unique(yvals))
  
  data$xgrid <- sapply(xvals, function(x) which(x == xvals_u))
  data$ygrid <- sapply(yvals, function(x) which(x == yvals_u))
  
  p <- data %>%
    ggplot(aes(xgrid, ygrid, 
               fill = log( eval(parse(text=fillvar)) ))) +
    geom_raster() +
    scale_x_continuous(breaks = seq(length(xvals_u)), labels = xvals_u) +
    scale_y_continuous(breaks = seq(length(yvals_u)), labels = yvals_u) +
    scale_fill_continuous(type = 'viridis') +
    theme_minimal() +
    theme(panel.grid.minor = element_blank()) +
    labs(x = xlab, y = ylab,
         fill = paste0('log(', fill_lab, ')'))
  
  return(p)
}


gridsearchEDAandClean <- function(model, loss, lee2018 = FALSE) {
  
  if (lee2018 == TRUE) {
    # Investigate and remove NNs that failed fitting
    # Compare learning rate density between NN setups that trained
    #  and those that ever had loss of NA
    had_na <- loss %>%
      group_by(model_num) %>%
      filter(any(is.na(loss))) %>% 
      pull(model_num) %>% unique()
    # par(mfrow = c(1, 2))
    # ## No missing loss models
    # model %>%
    #   filter(!(model_num %in% had_na)) %>%
    #   pull(learning_rate) %>%
    #   log() %>% density() %>% plot(main = '')
    # ## Missing loss models
    # model %>%
    #   filter(model_num %in% had_na) %>%
    #   pull(learning_rate) %>%
    #   log() %>% density() %>% plot(main = '', ylab = '')
    # par(mfrow = c(1, 1))
    # mtext("Learning Rate Distribution:\nSuccessful vs. Failed NN Training", 
    #       side = 3, line = 1, outer = FALSE)
    # readline(prompt = paste0('Learning rate distribution for NNs that',
    #                          ' successfully trained vs. failed training.\n',
    #                          'Press [Enter] to continue:'))
    # # Conclusion: likely due to exploding gradients; exclude
    # #  model numbers that didn't train successfully.min_loss <- loss %>%
    loss <- loss %>%
      filter(!(model_num %in% had_na))
  }
  
  min_loss <- loss %>%
    group_by(model_num) %>%
    summarize(min_loss = min(loss)) %>%
    inner_join(model, by = 'model_num')
  
  # Compare average best loss for each group (layer_width, n_layers) of NNs
  avg_loss <- min_loss %>%
    group_by(layer_width, n_layers) %>%
    summarize(min_loss = mean(min_loss))
  customRaster(avg_loss, xvar = 'layer_width', yvar = 'n_layers', 
               fillvar = 'min_loss', xlab = 'Layer Width', 
               ylab = 'Hidden Layers', fill_lab = 'LOSS') %>%
    print()
  readline(prompt = paste0('Log average of best validation LOSS for ',
                           'each trained NN, grouped by (Layer Width, # ',
                           'of Hidden Layers).\n',
                           'Press [Enter] to continue:'))
  # Does batch size affect the LOSS performance?
  p <- min_loss %>%
    ggplot(aes(as.factor(batch_size), log(min_loss))) + 
    geom_violin() + 
    labs(x = 'Batch Size', y = 'log(Best LOSS)') +
    coord_flip()
  print(p)
  readline(prompt = paste0('Log average of best validation LOSS for ',
                           'each trained NN, grouped by (Batch Size, ',
                           'Layer Width).\n',
                           'Press [Enter] to continue:'))
  
  if (lee2018 == TRUE) {
    # # Does weight decay make a difference in minimum val LOSS?
    # min_loss %>%
    #   ggplot(aes(weight_decay, log(min_loss))) +
    #   geom_point() +
    #   labs(x = 'Weight Decay', y = 'Log(Best LOSS)')
    # 
    # # How about learning rate?
    # min_loss %>%
    #   ggplot(aes(learning_rate, log(min_loss))) +
    #   geom_point(alpha = 0.7) +
    #   labs(x = 'Learning Rate', y = 'log(Best LOSS)')
    
    # Interesting effect on LOSS for (learning rate, weight decay) combos
    # Not visible in raster plot
    # min_loss %>%
    #   mutate(log_learning_rate = round(log(learning_rate)),
    #          log_weight_decay = round(log(weight_decay)/2)) %>%
    #   customRaster(xvar = 'log_learning_rate', yvar = 'log_weight_decay', 
    #                fillvar = 'min_loss', xlab = 'log(Learning Rate)', 
    #                ylab = 'log(Weight Decay)', fill_lab = 'LOSS')
    p <- min_loss %>%
      ggplot(aes(log(learning_rate), log(weight_decay), col = log(min_loss))) +
      geom_point(alpha = 0.9, size = 2) +
      scale_color_continuous(type = 'viridis') +
      labs(x = 'log(Learning Rate)', y = 'log(Weight Decay)', 
           col = 'log(Best LOSS)')
    print(p)
    readline(prompt = paste0('Log LOSS for each trained NN across ',
                             'varying learning rate and weight decay settings.\n',
                             'Press [Enter] to continue:'))
  }
  
  # Filter down to LOSS less than 5
  p <- loss %>%
    ggplot(aes(x = epoch, y = loss, group = model_num)) +
    geom_line(alpha = 0.5) +
    facet_grid(n_layers ~ layer_width) +
    scale_y_continuous(limits = c(NA, 5)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(x = 'Epoch')
  print(p)
  readline(prompt = paste0('LOSS for all trained NN.\n',
                           'Press [Enter] to continue:'))
  
  ### 5 Best Models: Lowest LOSS Anywhere -------------------------------------
  
  # Filter down to models with lowest loss at any epoch
  min_loss_modelnum <- min_loss %>%
    # group_by(model_num) %>%
    # summarize(min_loss = min(loss)) %>%
    slice_min(min_loss, n = 5) %>%
    pull(model_num)
  loss_best <- loss %>%
    filter(model_num %in% min_loss_modelnum)
  
  p <- loss_best %>%
    ggplot(aes(x = epoch, y = loss, 
               group = model_num, 
               col = as.factor(model_num))) +
    geom_line(alpha = 0.3) +
    geom_smooth(se = FALSE) +
    labs(col = 'Model', x = 'Epoch', y = 'LOSS') +
    theme_bw() +
    scale_y_continuous(limits = c(NA, 2.75))
  print(p)
  readline(prompt = paste0('Validation LOSS for 5 NN with lowest LOSS.\n',
                           'Press [Enter] to continue:'))
  
  best5 <- loss_best %>%
    group_by(model_num) %>%
    summarize(min_loss = round(min(loss), 4),
              min_loss_epoch = which.min(loss)) %>%
    inner_join(model, by = 'model_num') %>%
    select(-c(min_loss, min_loss_epoch),
           c(min_loss, min_loss_epoch))
  
  return(best5)
  
}