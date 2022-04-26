##
## Spatial Basis for Spatial Categories
##

## Libraries
library(tidyverse)
library(keras)
source('MakeNNModel.R')
# library(RSpectra)

# set working directory if necessary
tryCatch(setwd("C:/Users/skyle/Dropbox/2021 Fall/research/Quant150K"),
         error = function(cond) {
           message(paste0('Could not set directory. ',
                          'Assuming code is being run via Bash.'))
         })
# Read in data
load('data/AllSimulatedTemps.RData')
data_train <- all.sim.data %>%
  rename(x = Lon, y = Lat) %>%
  filter(!is.na(MaskTemp))
data_test <- all.sim.data %>%
  rename(x = Lon, y = Lat) %>%
  filter(is.na(MaskTemp))
rm(all.sim.data)

## Calculate Hughes & Haran Basis Functions
# A <- ngspatial::adjacency.matrix(size)
# P.orthog <- -(1/(size^2))*matrix(1, nrow=size^2, ncol=size^2)
# diag(P.orthog) <- 1-diag(P.orthog)
# M <- (P.orthog%*%A%*%P.orthog) %>% eigs_sym(., which="LA", k=250)
# sub.df2 <- bind_cols(sub.df, as.data.frame(M$vectors))
# ggplot(data=sub.df2, aes(x=x, y=y, fill=V250)) + 
#   geom_raster()


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

theta <- 1.5 * min(offset_x * 2, offset_y * 2)
D <- fields::rdist(data_train[, c('x', 'y')], bases) / theta
X <- wendland(D)
X[D > 1] <- 0
rm(D)

colnames(X) <- paste0("X", 1:ncol(X))
data_train2 <- bind_cols(data_train, as.data.frame(X))
rm(X)

# Observe one of the bases
# data_train2 %>%
data_train2 %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(basis = mean(X1)) %>%
  ggplot(aes(x, y, fill = basis)) +
  geom_raster()

# Fit a neural network to bases -------------------------------------------


x_train <- data_train2 %>%
  select(-c(MaskTemp:TrueTemp)) %>%
  as.matrix.data.frame()
y_train <- data_train2 %>%
  select(MaskTemp) %>%
  as.matrix.data.frame()
# x_test <- data_test %>%
#   as.matrix.data.frame()

model_pars <- c(n_layers = 3, layer_width = 2^6,
                epochs = 20, batch_size = 2^7,
                decay_rate = 0, square_feat = 0,
                dropout_rate = 0, model_num = 1,
                INPUT_SHAPE = ncol(x_train))

set.seed(1812)
model <- fitModel(model_pars, x_train, y_train)

# Predictions after fitting 80% of training set
predicted <- model %>%
  predict(x_train)

data_pred <- cbind(data_train, predicted) %>%
  pivot_longer(cols = c(values, predicted),
               names_to = 'type',
               values_to = 'z')

simple_train <- data_pred %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y, type) %>%
  summarize(z = mean(z))

simple_train %>%
  ggplot(aes(x, y, fill = z)) +
  geom_raster() +
  facet_wrap(~type)







## Write out the dataset
# write.csv(x=data_train2, file="./SkylerData400Basis.csv", row.names=FALSE)