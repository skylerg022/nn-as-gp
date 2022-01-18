

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
