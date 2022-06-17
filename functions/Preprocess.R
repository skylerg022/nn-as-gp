

# Multi-resolution spatial basis function expansion
# Inputs:
# - x_train: matrix of all training location data
# - x_withheld: matrix of all withheld location data
# - sqrt_n_knots: a vector of integers. Larger integers means a
#     finer grid of basis function expansions.
# - local_n: a numeric value for the minimum local sample size required to return
#     a basis/knot as part of the radial basis function expansion.
# - closest_minval: a numeric value between 0 and 1 stipulating how close the
#     closest observation to each basis/knot must be in order to keep that
#     basis/knot as part of the returned radial basis function expansion. A
#     higher value translates the closest observation being nearer to the knot.
# - test: if TRUE, will return transformed withheld x's as x_test, x_val
#     otherwise. This argument is used chiefly to conserve RAM in parallel
#     processing during grid searches.
# Output:
# - A named list of two matrices of dimensions nxk and mxk, where n and m are the 
#     number of observations in x_train and x_withheld, respectively, and k is
#     the number of basis transformations with a large enough local sample size.
#     k is sum(sqrt_n_knots^2) at most.
multiResBases <- function(x_train, x_withheld, sqrt_n_knots, 
                          local_n, closest_minval, test = TRUE) {
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
    basis_n <- colSums(X)
    max_val <- apply(X, 2, max)
    good_bases_idx <- which( (basis_n >= local_n) & (max_val > closest_minval) )
    
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
#  using all observations, let x_withheld be NULL.
# Inputs:
# - x_train: An n x p matrix of n observations and p numeric predictors.
# - x_withheld: An m x p matrix of m observations and p numeric predictors.
#     These observations are not used to calculate column means and
#     standard deviations.
# - test: if TRUE, will return transformed withheld x's as x_test, x_val
#     otherwise. This argument is used chiefly to conserve RAM in parallel
#     processing during grid searches.
# Output:
# - A 2-item list with x_train and x_withheld matrices transformed using only
#     data from the x_train matrix.
predictorsScaled <- function(x_train, x_withheld = NULL, test = TRUE) {
  n_train <- nrow(x_train)
  n_withheld <- nrow(x_withheld)
  
  mean_train <- colMeans(x_train)
  sd_train <- apply(x_train, 2, sd)
  
  # Function to help with matrix math
  quickMatrix <- function(n, x) {
    matrix(x, ncol = length(x), nrow = n, byrow = TRUE)
  }
  
  # Scale predictors with appropriate statistics
  x_train <- ( (x_train - quickMatrix(n_train, mean_train)) /
                 quickMatrix(n_train, sd_train) ) %>%
    as.matrix()
  if(!is.null(x_withheld)) {
    x_withheld <- ( (x_withheld - quickMatrix(n_withheld, mean_train)) /
                  quickMatrix(n_withheld, sd_train) ) %>%
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


# discretize
# Create evenly spaced continuous values accross the location space while 
#  maintaining the approximate range and value of observation values.
# Inputs:
# - x: A vector of numeric values.
# - nbins: an integer representing the number of evenly spaced values to bin x into.
# Output:
# - A vector of numeric values the same length as x with values evenly spaced
#     distance between bins
discretize <- function(x, nbins = 100) {
  # If already evenly spaced, don't "discretize" (Consider coding this up)
  
  # otherwise... make discrete
  binvals <- seq(min(x), max(x), length.out = nbins-1)
  bindist <- binvals[2] - binvals[1]
  edges <- c(binvals - bindist/2, max(x) + bindist/2)
  bin_idx <- .bincode(x, breaks = edges)
  return(binvals[bin_idx])
}
