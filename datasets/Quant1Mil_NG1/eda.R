## Skyler Gray
## Exploratory data analysis of quant900k training data

## Libraries
library(tidyverse)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../HelperFunctions/Defaults.R')

# Set seed
set.seed(22122)

# Read in data
data_train <- read.csv('data/dataset2_training.csv')
data_test <- read.csv('data/dataset2_testing.csv')

# EDA ---------------------------------------------------------------------

# How is the spatial data distributed?
data_train %>%
  ggplot(aes(x, y)) +
  geom_bin_2d()
# The data is approximately distributed uniformly across x and y.

# How about for test data?
data_test %>%
  ggplot(aes(x, y)) +
  geom_bin_2d()
# Approximately uniform across both axes as well

# Are there multiple observations in one location?
ndistinct <- data_train %>%
  bind_rows(data_test) %>%
  select(x, y) %>%
  distinct() %>%
  nrow()
ndistinct / ( nrow(data_train) + nrow(data_test) )
# All observations in the combined training and test data are distinct
#  and have a unique location

# Correlation looks apparent. Show variogram
# vario <- variog(coords = cbind(simple_train$x, simple_train$y),
#                 data = simple_train$values)
# plot(vario)
# Correlation present

# Visualize data
# points not evenly spaced from each other. Averaging values in each "bin"
# and displaying a 100 x 100 representation of the spatial data
data_train %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(Value = mean(values)) %>%
  ggplot(aes(x, y, fill = Value)) +
  geom_raster() +
  scale_fill_continuous(type = 'viridis') +
  theme_minimal()

ggsave('pics/train.png',
       width = pic_width * (2/3),
       height = pic_height,
       units = pic_units,
       bg = 'white')

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
     x_test, 
     file = 'data/dataset2_split.RData')
