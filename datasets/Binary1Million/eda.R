## Exploratory data analysis of Binary1Mil data

## Libraries
library(tidyverse)
library(raster)
library(RSpectra)
library(colorspace)

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

# Set seed
set.seed(425)

# Read in the Data
file.name <- "data/grf_x10_y10_vf0.1_seed1.tif"
dat <- raster(file.name) %>%
  as.matrix()

# Convert to dataframe and plot
dat.df <- data.frame(x = rep(1:ncol(dat), nrow(dat)),
                     y = rep(1:nrow(dat), each = ncol(dat)),
                     z = c(dat))
p <- dat.df %>%
  ggplot(aes(x = x, y = y, fill = as.factor(z))) + 
  geom_raster() +
  scale_fill_manual(values = qualitative_hcl(2, c = 100, l = 70)) +
  labs(fill = 'value')
p
myggsave(filename = 'pics/all.png', plot = p)
myggsave(filename = 'pics/all.pdf', plot = p)

# Data from a raster file is uniformly distributed across the 2D space

# Correlation apparent. Too large a sample size to view variogram.

# Randomly assign train, val, and test data
N <- nrow(dat.df)
n_test <- 100000
n_train <- N - n_test
test_idx <- sample(1:N, n_test, replace = FALSE)
val_idx <- sample(1:n_train, n_train * (0.2), replace = FALSE)


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
     file = 'data/DataSplit.RData')
