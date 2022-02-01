

# Default variables for saving pics
pic_width <- 8
pic_height <- 4
pic_units <- 'in'

dirCheck <- function() {
  # if (!dir.exists('data')) dir.create('data')
  if (!dir.exists('data/gridsearch_nn')) dir.create('data/gridsearch_nn')
  if (!dir.exists('data/gridsearch_nn_trans')) dir.create('data/gridsearch_nn_trans')
  if (!dir.exists('data/gridsearch_basis')) dir.create('data/gridsearch_basis')
  if (!dir.exists('pics')) dir.create('pics')
  return()
}

# Create grid of bases/knots ----------------------------------------------------

# Multi-resolution spatial basis function expansion
# Inputs:
# - x_train: matrix of all training data
# - sqrt_n_knots: a vector of integers. Larger integers means a
#     finer grid of basis function expansions.
# - min_n: an integer for the required minimum sample size for
#     a basis/knot to be kept for training
# Output:
# - Basis function expansion of various resolutions where all
#     bases have at least min_n non-zero values (n_min local observations)
multiResBases <- function(x_train, x_test, sqrt_n_knots, 
                          thresh_type, thresh, thresh_max) {
  ## Calculate Wendland basis
  wendland <- function(d){
    ((1-d)^6) * (35*d^2 + 18*d + 3) / 3
  }
  
  xmin <- min(x_train$x)
  xmax <- max(x_train$x)
  ymin <- min(x_train$y)
  ymax <- max(x_train$y)
  
  x_bases <- matrix(nrow = nrow(x_train), ncol = 0)
  x_bases_test <- matrix(nrow = nrow(x_test), ncol = 0)
  
  for (knots in sqrt_n_knots) {
    
    # Get basis grid X locations
    temp <- seq(xmin,
                xmax,
                length = knots + 1)
    offset_x <- (temp[2] - temp[1]) / 100
    # offset_x <- 0
    grid_x <- seq(xmin + offset_x,
                  xmax - offset_x,
                  length = knots)
    
    # Get basis grid Y locations
    temp <- seq(ymin,
                ymax,
                length = knots + 1)
    offset_y <- (temp[2] - temp[1]) / 100
    # offset_y <- 0
    grid_y <- seq(ymin + offset_y,
                  ymax - offset_y,
                  length = knots)
    
    bases <- expand.grid(grid_x, grid_y)
    # qplot(bases[,1], bases[,2], geom = 'point')
    
    # theta <- 1.5 * min(offset_x * 2, offset_y * 2)
    theta <- 2 * min((grid_x[2] - grid_x[1]), (grid_y[2] - grid_y[1]))
    
    D <- fields::rdist(x_train[, c('x', 'y')], bases) / theta
    X <- wendland(D)
    X[D > 1] <- 0
    rm(D)
    
    colnames(X) <- paste0("X", knots, '_', 1:ncol(X))
    
    # Assess which bases to keep
    if (threshold_type == 'nonzero') {
      n_nonzero <- colSums(X != 0)
      good_bases_idx <- which(n_nonzero >= threshold)
    } else if (threshold_type == 'colsum') {
      sums <- colSums(X)
      max_val <- apply(X, 2, max)
      good_bases_idx <- which( (sums >= threshold) & (max_val > thresh_max) )
    } else {
      err <- paste0("Threshold type not recognized. Currently, only 'nonzero' ", 
                    "and 'colsum' are allowed.")
      stop(err)
    }
    x_bases <- cbind(x_bases, X[,good_bases_idx])
    
    # Basis function expansion for x_test on training bases
    D <- fields::rdist(x_test[, c('x', 'y')], bases[good_bases_idx,]) / theta
    X <- wendland(D)
    X[D > 1] <- 0
    rm(D)
    x_bases_test <- cbind(x_bases_test, X)
    
    rm(X)
  }
  
  return(list(x_train = x_bases,
              x_test = x_bases_test))
}
