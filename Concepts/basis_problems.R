## An illustration of the problem of basis function expansions
##  applied to the Quant150K satellite data

## Libraries
library(tidyverse)
library(keras)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}
source('../HelperFunctions/MakeNNModel.R')
source('../HelperFunctions/Defaults.R')


# Read in data
load('../Quant150K_Sim/data/AllSimulatedTemps.RData')
data_train <- all.sim.data %>%
  rename(x = Lon, y = Lat) %>%
  filter(!is.na(MaskTemp)) %>%
  # Because blocks of data are in test dataset, validation set should be blocks
  mutate(validation = ifelse( (((x > -94.5 & x < -93) & (y < 34.75)) |
                                 ((x > -93.5 & x < -93) & (y > 35.5 & y < 36)) |
                                 ((x > -91.75) & (y > 35.25 & y < 35.75)) |
                                 ((x > -95.75 & x < -95.25) & (y > 36.25 & y < 36.75)) |
                                 ((x > -95 & x < -94.5) & (y > 35.25 & y < 35.75)) |
                                 ((x > -92.5 & x < -91.75) & (y > 36 & y < 36.75))),
                              1, 0)) %>%
  arrange(validation)
data_test <- all.sim.data %>%
  rename(x = Lon, y = Lat) %>%
  filter(is.na(MaskTemp))
rm(all.sim.data)

# Create grid of bases/knots ----------------------------------------------------

## Calculate Wendland basis
wendland <- function(d){
  ((1-d)^6) * (35*d^2 + 18*d + 3) / 3
}

xmin <- min(data_train$x)
xmax <- max(data_train$x)
ymin <- min(data_train$y)
ymax <- max(data_train$y)
n_parts <- 20

# Get basis grid X locations
temp <- seq(xmin, 
            xmax, 
            length = n_parts + 1)
offset_x <- (temp[2] - temp[1]) / 2
grid_x <- seq(xmin + offset_x, 
              xmax - offset_x, 
              length = n_parts)

# Get basis grid Y locations
temp <- seq(ymin, 
            ymax, 
            length = n_parts + 1)
offset_y <- (temp[2] - temp[1]) / 2
grid_y <- seq(ymin + offset_y, 
              ymax - offset_y, 
              length = n_parts)

bases <- expand.grid(grid_x, grid_y)
# qplot(bases[,1], bases[,2], geom = 'point')

# theta <- 1.5 * min(offset_x * 2, offset_y * 2)
theta <- 1.5 * min(offset_x * 2, offset_y * 2)
D <- fields::rdist(data_train[, c('x', 'y')], bases) / theta
X <- wendland(D)
X[D > 1] <- 0
rm(D)

colnames(X) <- paste0("X", 1:ncol(X))



# Analysis ----------------------------------------------------------------

# Ranges in each column are no more than from 0 to 1
apply(X, 2, min) %>% table()
apply(X, 2, max) %>% hist()

# The count of nonzero values in each column represent the
#  sample size to train NN weights that rely on that column
n_nonzero <- colSums(X != 0)
hist(n_nonzero)

# Identify bases that may cause problems with fitting
bases_cols <- ifelse(n_nonzero < 500, 'red', 'black')

bases_ggplot <- data.frame(x = bases[,1],
                           y = bases[,2],
                           N_Nonzero = n_nonzero)

# Visualize data
# points not evenly spaced from each other. Averaging values in each "bin"
# and displaying a 100 x 100 representation of the spatial data
data_train %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(Temp = mean(MaskTemp)) %>%
  ggplot(aes(x, y, fill = Temp)) +
  geom_raster() +
  geom_point(data = bases_ggplot,
             mapping = aes(x=x, y=y, col = N_Nonzero,
                           fill = 30),
             size = 2) + # Manually coded this fill value to avoid error
  scale_fill_continuous(type = 'viridis') +
  scale_color_steps2(midpoint = 800,
                     low = 'red') +
  theme_minimal() +
  labs(col = 'Local N')

