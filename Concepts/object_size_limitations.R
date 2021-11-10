library(pryr)
library(ggplot2)

## General concept: As size of square matrix increases linearly,
##  object size increases quadratically
k <- 1:1000
size <- sapply(k, function(k) object_size(matrix(0, nrow = k, ncol = k)))
qplot(k, size, geom = 'line')

## The maximum number of bases for expansion is dependent on
##  the number of observations and the size of RAM in bytes
maxBases <- function(ram, n_obs) {
  n_matrix_cells <- ram / 8
  # Calculate max bases
  n_bases <- n_matrix_cells %/% n_obs
  # Divide bases out on a 2D grid. How many breaks are allowed per dimension?
  n_axis_breaks <- round(sqrt(n_bases))
  cat(sprintf('Max number of spatial bases:    %.0f\n', n_bases),
      sprintf('Number of breaks per axis (2D): %.0f\n\n',
              n_axis_breaks),
      sep = '')
}

# Case: Quant 150K
# RAM: 1e9 Bytes = 1 GB
maxBases(ram = 1e9, 105569)
