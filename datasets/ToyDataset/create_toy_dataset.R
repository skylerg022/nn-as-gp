
## Libraries
library(tidyverse)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

# Load default plot saving function and create pic/ directories
source('../../functions/Defaults.R')
# Load discretize() function
source('../../functions/Preprocess.R')
CheckDir()

# Set seed
set.seed(5422)

# Read in data and filter down to about 9000 observations
data_train <- read.csv('../Quant1Mil_NG1/data/dataset2_training.csv') %>%
  filter(x > 0.4 & x <= 0.5,
         y > 0.1 & y <= 0.2)

# Randomly assign observations to test set
n_train <- nrow(data_train)
test_idx <- sample(1:n_train, size = 900, replace = FALSE)

data_test <- data_train[test_idx,]
data_train <- data_train[-test_idx,]

save(data_train, data_test,
     file = 'data/ToyData.RData')
