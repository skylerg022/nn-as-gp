
# Load libraries
library(tidyverse)
library(keras)
library(parallel)

# Default variables for saving pics
pic_width <- 8
pic_height <- 4
pic_units <- 'in'

dirCheck <- function() {
  # if (!dir.exists('data')) dir.create('data')
  if (!dir.exists('pics')) {
    message('No existing pic/ directory detected. Creating pic/ directory...')
    dir.create('pics')
    }
  return(NULL)
}

# Set default ggplot theme
theme_set(theme_minimal())