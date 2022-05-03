
# Note: also relies on Defaults.R. Make sure to load this as well within your code

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
  write_csv(grid, paste0('data/gridsearch/grid_', input_type, '_', 
                         basis_knots, modeltype, '.csv'))
  write_csv(val_df, paste0('data/gridsearch/grid_', input_type, '_', 
                           basis_knots, modeltype, '_val_loss.csv'))
  
  return()
}



# Post-search EDA Functions ------------------------------------------------

# Create a raster for all combinations of xvar and yvar hyperparameters
customRaster <- function(data, xvar, yvar, fillvar,
                         xlab, ylab,
                         binary_data = FALSE) {
  xvals <- pull(data, xvar)
  xvals_u <- sort(unique(xvals))
  yvals <- pull(data, yvar)
  yvals_u <- sort(unique(yvals))
  
  data$xgrid <- sapply(xvals, function(x) which(x == xvals_u))
  data$ygrid <- sapply(yvals, function(x) which(x == yvals_u))
  
  if (binary_data == TRUE) {
    Trans <- function(x) x
    fill_lab <- 'Min(Cross-\n  entropy)'
  } else {
    Trans <- function(x) log(x)
    fill_lab <- 'Min(log(RMSE))'
  }
  
  p <- data %>%
    ggplot(aes(xgrid, ygrid, 
               fill = Trans( eval(parse(text=fillvar)) ))) +
    geom_raster() +
    scale_x_continuous(breaks = seq(length(xvals_u)), labels = xvals_u) +
    scale_y_continuous(breaks = seq(length(yvals_u)), labels = yvals_u) +
    scale_fill_continuous(type = 'viridis') +
    theme_minimal() +
    theme(panel.grid.minor = element_blank()) +
    labs(x = xlab, y = ylab,
         fill = fill_lab)
  
  return(p)
}


gridsearchEDAandClean <- function(model, loss, 
                                  pars_type,
                                  binary_data = FALSE,
                                  filename = '') {
  
  if (pars_type == 'lee2018') {
    # If learning rate is too high, NNs sometimes fail fitting.
    #   Removing NNs that ever had loss of NA
    had_na <- loss %>%
      group_by(model_num) %>%
      filter(any(is.na(loss))) %>% 
      pull(model_num) %>% 
      unique()
    loss <- loss %>%
      filter(!(model_num %in% had_na))
  }
  
  min_loss <- loss %>%
    group_by(model_num) %>%
    summarize(min_loss = signif( min(loss), 3 )) %>%
    inner_join(model, by = 'model_num')
  
  # Compare average best loss for each group (layer_width, n_layers) of NNs
  vis_min_loss <- min_loss %>%
    group_by(layer_width, n_layers) %>%
    summarize(min_loss = min(min_loss))
  p <- customRaster(vis_min_loss, xvar = 'layer_width', yvar = 'n_layers', 
                    fillvar = 'min_loss', xlab = 'Layer Width', 
                    ylab = 'Hidden Layers',
                    binary_data = binary_data)
  myggsave(filename = paste0('pics/gridsearch/frame_', filename, '.pdf'), plot = p)
  
  ### Best Models: Lowest LOSS -------------------------------------
  # Looking for lowest loss (to 3 sig figs) and largest number of epochs trained
  #  (to act as a tie-breaker for models with the same loss)
  
  # Filter down to models with lowest loss at any epoch
  min_loss_modelnum <- min_loss %>%
    filter(min_loss == min(min_loss)) %>%
    pull(model_num)
  loss_best <- loss %>%
    filter(model_num %in% min_loss_modelnum)
  
  best <- loss_best %>%
    group_by(model_num) %>%
    summarize(min_loss = signif(min(loss), 3),
              min_loss_epoch = which.min(loss)) %>%
    ungroup() %>%
    filter(min_loss_epoch == max(min_loss_epoch)) %>%
    inner_join(model, by = 'model_num') %>%
    select(-c(min_loss, min_loss_epoch),
           c(min_loss, min_loss_epoch))
  
  return(best)
}
