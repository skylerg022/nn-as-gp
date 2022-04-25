## Skyler Gray
## Exploratory data analysis of Binary1Mil data


## Libraries
library(tidyverse)
library(raster)
library(RSpectra)
library(colorspace)

theme_set(theme_minimal())

# Set seed
set.seed(425)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../HelperFunctions/Defaults.R')

## Read in the Data
file.name <- "data/grf_x10_y10_vf0.1_seed1.tif"
dat <- raster(file.name) %>%
  as.matrix()

# Convert to dataframe and plot
dat.df <- data.frame(x = rep(1:ncol(dat), nrow(dat)),
                     y = rep(1:nrow(dat), each = ncol(dat)),
                     z = as.factor(c(dat)))
ggplot(data = dat.df, aes(x = x, y = y, fill = z)) + 
  geom_raster() +
  scale_fill_manual(values = qualitative_hcl(2, c = 100, l = 70)) +
  labs(fill = 'value')

ggsave('pics/all.png',
       width = pic_width * (2/3),
       height = pic_height,
       units = pic_units,
       bg = 'white')

# Data from a raster file is uniformly distributed across the 2D space

# Correlation apparent. Too large a sample size to view variogram.


# Randomly assign train, val, and test data
N <- nrow(dat.df)
n_test <- 100000
n_train <- N - n_test
test_idx <- sample(1:N, n_test, replace = FALSE)
val_idx <- sample(1:n_train, n_train * (0.2), replace = FALSE)

dat.df[-test_idx,][-val_idx,] %>% nrow()

# Where is test set located?
data_test %>%
  mutate(across(c(x,y), round, digits = 3)) %>%
  ggplot(aes(x, y)) +
  geom_raster(fill = 'black') +
  theme_minimal()
# No highly concentrated regions

# Saving data -------------------------------------------------------------

# Train
x_train <- dat.df[-test_idx,][-val_idx, c(1,2)] %>%
  as.matrix()
y_train <- dat.df[-test_idx,][-val_idx, 3, drop = FALSE] %>%
  as.matrix()

# Validation
x_val <- dat.df[-test_idx,][val_idx, c(1,2)] %>%
  as.matrix()
y_val <- dat.df[-test_idx,][val_idx, 3, drop = FALSE] %>%
  as.matrix()

# Test
x_test <- dat.df[test_idx, c(1,2)] %>%
  as.matrix()
y_test <- dat.df[test_idx, 3, drop = FALSE] %>%
  as.matrix()

# Save
save(x_train, y_train,
     x_val, y_val,
     x_test, y_test,
     file = 'data/seed1_split.RData')
