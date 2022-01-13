## Misc. helper functions

# predictorStats
# Calculate mean and sd of each predictor, withholding validation set
predictorsScaled <- function(x_train, val_split = 0.2) {
  n <- nrow(x_train)
  n_train <- floor(nrow(x_train) * (1-val_split))
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
  
  return(x_train)
}




