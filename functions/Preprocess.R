

# Multi-resolution spatial basis function expansion
# Inputs:
# - x_train: matrix of all training data
# - x_withheld: matrix of all withheld data
# - sqrt_n_knots: a vector of integers. Larger integers means a
#     finer grid of basis function expansions.
# - min_n: an integer for the required minimum sample size for
#     a basis/knot to be kept for training
# - test: if TRUE, will return transformed, withheld x's as x_test, x_val otherwise
# Output:
# - Basis function expansion of various resolutions where all
#     bases have at least min_n non-zero values (n_min local observations)
multiResBases <- function(x_train, x_withheld, sqrt_n_knots, 
                          thresh_type, thresh, thresh_max,
                          test = TRUE) {
  ## Calculate Wendland basis
  wendland <- function(d){
    ((1-d)^6) * (35*d^2 + 18*d + 3) / 3
  }
  
  xmin <- min(x_train[,'x'])
  xmax <- max(x_train[,'x'])
  ymin <- min(x_train[,'y'])
  ymax <- max(x_train[,'y'])
  
  x_bases <- matrix(nrow = nrow(x_train), ncol = 0)
  x_bases_withheld <- matrix(nrow = nrow(x_withheld), ncol = 0)
  
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
    if (thresh_type == 'nonzero') {
      n_nonzero <- colSums(X != 0)
      good_bases_idx <- which(n_nonzero >= thresh)
    } else if (thresh_type == 'colsum') {
      sums <- colSums(X)
      max_val <- apply(X, 2, max)
      good_bases_idx <- which( (sums >= thresh) & (max_val > thresh_max) )
    } else {
      err <- paste0("Threshold type not recognized. Currently, only 'nonzero' ", 
                    "and 'colsum' are allowed.")
      stop(err)
    }
    x_bases <- cbind(x_bases, X[,good_bases_idx])
    
    # Basis function expansion for x_withheld on training bases
    D <- fields::rdist(x_withheld[, c('x', 'y')], bases[good_bases_idx,]) / theta
    X <- wendland(D)
    X[D > 1] <- 0
    rm(D)
    x_bases_withheld <- cbind(x_bases_withheld, X)
    
    rm(X)
  }
  
  if (test == TRUE) {
    return(list(x_train = x_bases,
                x_test = x_bases_withheld))
  } else {
    return(list(x_train = x_bases,
                x_val = x_bases_withheld))
  }
}


# predictorsScaled
# Center and scale all observations by columns. To scale the data
#  using all observations, let x_withheld be NULL and set val_split to 0.
#  When x_withheld is not null, val_split is ignored.
# Inputs:
# - x_train: An n x p matrix of n observations and p numeric predictors.
# - x_withheld: An m x p matrix of m observations and p numeric predictors.
#     These observations will never be used to calculate column means and
#     standard deviations.
# - val_split: the proportion of training data to exclude (last 100*val_split 
#     percent of the observations) when calculating column means and standard
#     devations.
# Output:
# - An (n+m) x p matrix of all observations, centered and scaled using
#     just the training data.
predictorsScaled <- function(x_train, x_withheld = NULL, val_split = 0.2, test = TRUE) {
  n <- nrow(x_train)
  if (is.null(x_withheld)) {
    n_train <- floor(nrow(x_train) * (1-val_split))
  } else {
    n_train <- n
  }
  mean_train <- x_train[1:n_train,] %>%
    colMeans()
  sd_train <- x_train[1:n_train,] %>%
    apply(2, sd)
  
  # Function to help with matrix math
  quickMatrix <- function(n, x) {
    matrix(x, ncol = length(x), nrow = n, byrow = TRUE)
  }
  
  # Scale predictors with appropriate statistics
  x_train <- ( (x_train - quickMatrix(n, mean_train)) /
                 quickMatrix(n, sd_train) ) %>%
    as.matrix()
  if (!is.null(x_withheld)) {
    x_withheld <- ( (x_withheld - quickMatrix(nrow(x_withheld), mean_train)) /
                  quickMatrix(nrow(x_withheld), sd_train) ) %>%
      as.matrix()
  }
  
  if (test == TRUE) {
    return(list(x_train = x_train,
                x_test = x_withheld))
  } else {
    return(list(x_train = x_train,
                x_val = x_withheld))
  }
}


# Create evenly spaced continuous values while maintaining the approximate 
#  range and value of observation values
# May not preserve range if number of bins is less than 100
discretize <- function(x, nbins = 100) {
  # If already evenly spaced, don't "discretize"
  
  # otherwise... make discrete
  edges <- seq(min(x), max(x), length.out = nbins + 1)
  binvals <- (edges[-1] + edges[1:nbins]) / 2
  bin_idx <- .bincode(x, breaks = edges, right = FALSE, include.lowest = TRUE)
  return(binvals[bin_idx])
}
