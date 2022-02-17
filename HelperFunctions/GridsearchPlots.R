

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
    #  and those that ever had rmse of NA
    had_na <- loss %>%
      group_by(model_num) %>%
      filter(any(is.na(rmse))) %>% 
      pull(model_num) %>% unique()
    par(mfrow = c(1, 2))
    ## No missing RMSE models
    model %>%
      filter(!(model_num %in% had_na)) %>%
      pull(learning_rate) %>%
      log() %>% density() %>% plot(main = '')
    ## Missing RMSE models
    model %>%
      filter(model_num %in% had_na) %>%
      pull(learning_rate) %>%
      log() %>% density() %>% plot(main = '', ylab = '')
    par(mfrow = c(1, 1))
    mtext("Learning Rate Distribution:\nSuccessful vs. Failed NN Training", 
          side = 3, line = 1, outer = FALSE)
    readline(prompt = paste0('Learning rate distribution for NNs that',
                             ' successfully trained vs. failed training.\n',
                             'Press [Enter] to continue:'))
    # Conclusion: likely due to exploding gradients; exclude
    #  model numbers that didn't train successfully.min_loss <- loss %>%
    loss <- loss %>%
      filter(!(model_num %in% had_na))
  }
  
  min_loss <- loss %>%
    group_by(model_num) %>%
    summarize(min_rmse = min(rmse)) %>%
    inner_join(model, by = 'model_num')
  
  # Compare average best RMSE for each group (layer_width, n_layers) of NNs
  avg_loss <- min_loss %>%
    group_by(layer_width, n_layers) %>%
    summarize(min_rmse = mean(min_rmse))
  customRaster(avg_loss, xvar = 'layer_width', yvar = 'n_layers', 
               fillvar = 'min_rmse', xlab = 'Layer Width', 
               ylab = 'Hidden Layers', fill_lab = 'RMSE') %>%
    print()
  readline(prompt = paste0('Log average of best validation RMSE for ',
                           'each trained NN, grouped by (Layer Width, # ',
                           'of Hidden Layers).\n',
                           'Press [Enter] to continue:'))
  # Does batch size affect the RMSE performance?
  p <- min_loss %>%
    ggplot(aes(as.factor(batch_size), log(min_rmse))) + 
    geom_violin() + 
    labs(x = 'Batch Size', y = 'log(Best RMSE)') +
    coord_flip()
  print(p)
  readline(prompt = paste0('Log average of best validation RMSE for ',
                           'each trained NN, grouped by (Batch Size, ',
                           'Layer Width).\n',
                           'Press [Enter] to continue:'))
  
  if (lee2018 == TRUE) {
    # # Does weight decay make a difference in minimum val RMSE?
    # min_loss %>%
    #   ggplot(aes(weight_decay, log(min_rmse))) +
    #   geom_point() +
    #   labs(x = 'Weight Decay', y = 'Log(Best RMSE)')
    # 
    # # How about learning rate?
    # min_loss %>%
    #   ggplot(aes(learning_rate, log(min_rmse))) +
    #   geom_point(alpha = 0.7) +
    #   labs(x = 'Learning Rate', y = 'log(Best RMSE)')
    
    # Interesting effect on RMSE for (learning rate, weight decay) combos
    # Not visible in raster plot
    # min_loss %>%
    #   mutate(log_learning_rate = round(log(learning_rate)),
    #          log_weight_decay = round(log(weight_decay)/2)) %>%
    #   customRaster(xvar = 'log_learning_rate', yvar = 'log_weight_decay', 
    #                fillvar = 'min_rmse', xlab = 'log(Learning Rate)', 
    #                ylab = 'log(Weight Decay)', fill_lab = 'RMSE')
    p <- min_loss %>%
      ggplot(aes(log(learning_rate), log(weight_decay), col = log(min_rmse))) +
      geom_point(alpha = 0.9, size = 2) +
      scale_color_continuous(type = 'viridis') +
      labs(x = 'log(Learning Rate)', y = 'log(Weight Decay)', 
           col = 'log(Best RMSE)')
    print(p)
    readline(prompt = paste0('Log RMSE for each trained NN across ',
                             'varying learning rate and weight decay settings.\n',
                             'Press [Enter] to continue:'))
  }
  
  # Filter down to RMSE less than 5
  p <- loss %>%
    ggplot(aes(x = epoch, y = rmse, group = model_num)) +
    geom_line(alpha = 0.5) +
    facet_grid(n_layers ~ layer_width) +
    scale_y_continuous(limits = c(NA, 5)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(x = 'Epoch')
  print(p)
  readline(prompt = paste0('RMSE for all trained NN.\n',
                           'Press [Enter] to continue:'))
  
  ### 5 Best Models: Lowest RMSE Anywhere -------------------------------------
  
  # Filter down to models with lowest loss at any epoch
  min_loss_modelnum <- min_loss %>%
    # group_by(model_num) %>%
    # summarize(min_rmse = min(rmse)) %>%
    slice_min(min_rmse, n = 5) %>%
    pull(model_num)
  loss_best <- loss %>%
    filter(model_num %in% min_loss_modelnum)
  
  p <- loss_best %>%
    ggplot(aes(x = epoch, y = rmse, 
               group = model_num, 
               col = as.factor(model_num))) +
    geom_line(alpha = 0.3) +
    geom_smooth(se = FALSE) +
    labs(col = 'Model', x = 'Epoch', y = 'RMSE') +
    theme_bw() +
    scale_y_continuous(limits = c(NA, 2.75))
  print(p)
  readline(prompt = paste0('Validation RMSE for 5 NN with lowest RMSE.\n',
                           'Press [Enter] to continue:'))
  
  best5 <- loss_best %>%
    group_by(model_num) %>%
    summarize(min_rmse = round(min(rmse), 2),
              min_rmse_epoch = which.min(rmse)) %>%
    inner_join(model, by = 'model_num') %>%
    select(-c(min_rmse, min_rmse_epoch),
           c(min_rmse, min_rmse_epoch))
  
  return(best5)
  
  # 5 Best Models: Lowest Ending RMSE -------------------------------------
  # best_end_modelnum <- loss %>%
  #   group_by(model_num) %>%
  #   filter(epoch == max(epoch)) %>%
  #   ungroup() %>%
  #   slice_min(rmse, n = 5) %>%
  #   pull(model_num)
  # loss %>%
  #   filter(model_num %in% best_end_modelnum) %>%
  #   ggplot(aes(x = epoch, y = rmse, col = as.factor(model_num))) +
  #   geom_line(alpha = 0.3) +
  #   geom_smooth(se = FALSE) +
  #   labs(col = 'Model', x = 'Epoch', y = 'RMSE') +
  #   theme_bw()
  # model %>%
  #   filter(model_num %in% best_end_modelnum) %>%
  #   View()
}
