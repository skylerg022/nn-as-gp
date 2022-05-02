
# Load libraries
library(tidyverse)
library(keras)
library(parallel)
library(colorspace)

# GGplot defaults
# Binary response plot colors
colors_binary <- qualitative_hcl(2, c = 100, l = 70)
# Set default ggplot theme
theme_set(theme_minimal())

# Default variables for saving pics
pic_width <- 8
pic_height <- 4
pic_units <- 'in'

myggsave <- function(filename, plot) {
  ggsave(filename, plot,
         height = pic_height,
         width = pic_width * 2/3,
         units = pic_units,
         bg = 'white')
}

CheckDir <- function() {
  # if (!dir.exists('data')) dir.create('data')
  if (!dir.exists('pics')) {
    message('No existing pic/ directory detected. Creating pic/ directory...')
    dir.create('pics')
  }
  if (!dir.exists('pics/gridsearch')) {
    message('No existing pic/gridsearch/ directory detected. Creating directory...')
    dir.create('pics/gridsearch')
  }
  if (!dir.exists('pics/final')) {
    message('No existing pic/final/ directory detected. Creating directory...')
    dir.create('pics/final')
  }
  return(NULL)
}
