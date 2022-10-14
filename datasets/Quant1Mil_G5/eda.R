## Exploratory data analysis of Quant1Mil-G5 data

## Libraries
library(tidyverse)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

# Load default plot saving function and create pic/ directories
source('../../functions/Defaults.R')
CheckDir()

# Adjust ggplot theme
theme_set(theme(panel.background = element_rect(fill = 'gray30'),
                panel.grid = element_blank()))

# Set seed for random validation set assignment
set.seed(31522)

# Read in data
data_train <- read.csv('data/dataset01_training.csv')
data_test <- read.csv('data/dataset01_testing.csv')
y_test <- read.csv('data/Z_01.csv', header = FALSE) %>%
  as.matrix()


# EDA ---------------------------------------------------------------------

# How is the spatial data distributed?
data_train %>%
  ggplot(aes(x, y)) +
  geom_bin_2d()
# The data is approximately distributed uniformly across x and y.

# How about for test data?
p <- data_test %>%
  ggplot(aes(x, y)) +
  geom_bin_2d()
# Approximately uniform across both axes as well
p
myggsave(filename = 'pics/test.png', plot = p)
myggsave(filename = 'pics/test.pdf', plot = p)


# Are there multiple observations in one location?
ndistinct <- data_train %>%
  bind_rows(data_test) %>%
  select(x, y) %>%
  distinct() %>%
  nrow()
ndistinct / ( nrow(data_train) + nrow(data_test) )
# All observations in the combined training and test data are distinct
#  and have a unique location

# Visualize data
# points not evenly spaced from each other. Averaging values in each "bin"
# and displaying a 100 x 100 representation of the spatial data
p <- data_train %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(Value = mean(values)) %>%
  ggplot(aes(x, y, fill = Value)) +
  geom_raster() +
  scale_fill_continuous(type = 'viridis')
p
myggsave(filename = 'pics/train.png', plot = p)
myggsave(filename = 'pics/train.pdf', plot = p)


# Where is test set located?
data_test %>%
  mutate(across(c(x,y), round, digits = 3)) %>%
  ggplot(aes(x, y)) +
  geom_raster(fill = 'black') +
  theme_minimal()
# No highly concentrated regions


# Choosing a validation set -----------------------------------------------

# Randomly assign observations to validation set
n_train <- nrow(data_train)
val_idx <- sample(1:n_train, n_train * (0.2), replace = FALSE)

# Saving data -------------------------------------------------------------

# Train
x_train <- data_train[-val_idx, c(1,2)] %>%
  as.matrix()
y_train <- data_train[-val_idx, 3, drop = FALSE] %>%
  as.matrix()

# Validation
x_val <- data_train[val_idx, c(1,2)] %>%
  as.matrix()
y_val <- data_train[val_idx, 3, drop = FALSE] %>%
  as.matrix()

# Test
x_test <- data_test[,c(1,2)] %>%
  as.matrix()

# Save
save(x_train, y_train,
     x_val, y_val,
     x_test, y_test,
     file = 'data/DataSplit.RData')
