

# Default variables for saving pics
pic_width <- 8
pic_height <- 4
pic_units <- 'in'

dirCheck <- function() {
  # if (!dir.exists('data')) dir.create('data')
  if (!dir.exists('pics')) dir.create('pics')
  return()
}

