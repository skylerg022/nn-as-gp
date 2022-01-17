##
## Spatial Basis Function Expansion
##

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
load('data/AllSimulatedTemps.RData')
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
data_train2 <- bind_cols(data_train, as.data.frame(X))
rm(X)

# Observe one of the bases
# data_train2 %>%
#   mutate(across(c(x,y), round, digits = 2)) %>%
#   group_by(x, y) %>%
#   summarize(basis = mean(X1)) %>%
#   ggplot(aes(x, y, fill = basis)) +
#   geom_raster()


# Fit a neural network to bases -------------------------------------------

x_train <-data_train2 %>%
  select(-c(x:validation)) %>%
  as.matrix()
y_train <- data_train2 %>%
  select(MaskTemp) %>%
  as.matrix()

model_pars <- c(n_layers = 1, layer_width = 2^11,
                epochs = 20, batch_size = 2^7,
                decay_rate = 0, dropout_rate = 0, 
                model_num = 1)

model <- fitModel(model_pars, x_train, y_train, test = 'part_train')

# What areas are solely validation data?
# data_train2 %>%
#   mutate(across(c(x,y), round, digits = 2),
#          training = ifelse(row_number()/n() < 0.8, 1, 0)) %>%
#   group_by(x, y) %>%
#   summarize(training = min(training)) %>%
#   mutate(Dataset = ifelse(training == 1, 'Training', 'Validation')) %>%
#   ggplot(aes(x, y, fill = Dataset)) +
#   geom_raster() +
#   scale_fill_brewer(type = 'qual', palette = 6)


# Predictions after fitting 80% of training set
Predicted <- model %>%
  predict(x_train)

data_pred <- cbind(data_train, Predicted) %>%
  # clip predictions to max/min of observed data
  mutate(Predicted = ifelse(Predicted > max(TrueTemp), max(TrueTemp), Predicted),
         Predicted = ifelse(Predicted < min(TrueTemp), min(TrueTemp), Predicted)) %>%
  pivot_longer(cols = c(TrueTemp, Predicted),
               names_to = 'type',
               values_to = 'z')

simple_train <- data_pred %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y, type) %>%
  summarize(Temp = mean(z))

simple_train %>%
  ggplot(aes(x, y, fill = Temp)) +
  geom_raster() +
  facet_wrap(~type) +
  scale_fill_continuous(type = 'viridis') +
  theme_minimal()


# Test Predictions --------------------------------------------------------

model <- fitModel(model_pars, x_train, y_train, test = 'all_train')

D <- fields::rdist(data_test[, c('x', 'y')], bases) / theta
X <- wendland(D)
X[D > 1] <- 0
rm(D)

colnames(X) <- paste0("X", 1:ncol(X))
data_test2 <- bind_cols(data_test, as.data.frame(X))
rm(X)

x_test <- data_test2 %>%
  select(-c(x:TrueTemp)) %>%
  as.matrix.data.frame()
y_test <- data_test2 %>%
  select(TrueTemp) %>%
  as.matrix.data.frame()

# Predictions (including test set) after fitting 100% of training set
Predicted <- model %>%
  predict(rbind(x_train, x_test))

data_pred <- cbind(rbind(data_train %>% select(-validation), 
                         data_test), 
                   Predicted) %>%
  pivot_longer(cols = c(TrueTemp, Predicted),
               names_to = 'type',
               values_to = 'z')

simple_train <- data_pred %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y, type) %>%
  summarize(z = mean(z))

# Just test performance
simple_train %>%
  ggplot(aes(x, y, fill = z)) +
  geom_raster() +
  geom_raster(data = data_train2,
              mapping = aes(x, y, fill = min(simple_train$z))) +
  facet_wrap(~type) +
  scale_fill_continuous(type = 'viridis') +
  theme_minimal()

# All data prediction
simple_train %>%
  ggplot(aes(x, y, fill = z)) +
  geom_raster() +
  facet_wrap(~type) +
  scale_fill_continuous(type = 'viridis') +
  theme_minimal()


# Evaluate test RMSE
yhat <- model %>%
  predict(x_test)
sqrt(mean( (y_test - yhat)^2 ))

# Evaluate train RMSE
yhat <- model %>%
  predict(x_train)
sqrt(mean( (y_train - yhat)^2 ))

# # Repeat predictions with randomized training set -------------------------
# 
# set.seed(1812)
# 
# x_train <- data_train2 %>%
#   select(-c(x:TrueTemp)) %>%
#   as.matrix.data.frame()
# y_train <- data_train2 %>%
#   select(MaskTemp) %>%
#   as.matrix.data.frame()
# 
# # Shuffle training dataset
# shuffle_idx <- sample(1:nrow(x_train))
# x_train <- x_train[shuffle_idx,]
# y_train <- y_train[shuffle_idx, , drop=FALSE]
# 
# model_pars <- c(n_layers = 3, layer_width = 2^6,
#                 epochs = 20, batch_size = 2^7,
#                 decay_rate = 0, square_feat = 0,
#                 dropout_rate = 0, model_num = 1,
#                 input_size = ncol(x_train))
# 
# model <- fitModel(model_pars, x_train, y_train, test = 'part_train')
# 
# # What observations are training/test
# # data_train2 %>%
# #   mutate(across(c(x,y), round, digits = 2),
# #          training = ifelse(row_number()/n() < 0.8, 1, 0)) %>%
# #   group_by(x, y) %>%
# #   summarize(training = max(training)) %>%
# #   mutate(Dataset = ifelse(training == 1, 'Training', 'Validation')) %>%
# #   ggplot(aes(x, y, fill = Dataset)) +
# #   geom_raster() +
# #   scale_fill_brewer(type = 'qual', palette = 6)
# 
# 
# # Predictions after fitting 80% of training set
# Predicted <- model %>%
#   predict(x_train)
# 
# data_pred <- cbind(data_train[shuffle_idx,], Predicted) %>%
#   pivot_longer(cols = c(MaskTemp, Predicted),
#                names_to = 'type',
#                values_to = 'z')
# 
# simple_train <- data_pred %>%
#   mutate(across(c(x,y), round, digits = 2)) %>%
#   group_by(x, y, type) %>%
#   summarize(z = mean(z))
# 
# simple_train %>%
#   ggplot(aes(x, y, fill = z)) +
#   geom_raster() +
#   facet_wrap(~type)
# 
# 
# 
# 
# # Test Predictions --------------------------------------------------------
# 
# model <- fitModel(model_pars, x_train, y_train, test = 'all_train')
# 
# D <- fields::rdist(data_test[, c('x', 'y')], bases) / theta
# X <- wendland(D)
# X[D > 1] <- 0
# rm(D)
# 
# colnames(X) <- paste0("X", 1:ncol(X))
# data_test2 <- bind_cols(data_test, as.data.frame(X))
# rm(X)
# 
# x_test <- data_test2 %>%
#   select(-c(x:TrueTemp)) %>%
#   as.matrix.data.frame()
# y_test <- data_test2 %>%
#   select(TrueTemp) %>%
#   as.matrix.data.frame()
# 
# # Predictions (including test set) after fitting 100% of training set
# Predicted <- model %>%
#   predict(rbind(x_train, x_test))
# 
# data_pred <- cbind(rbind(data_train[shuffle_idx,], data_test), Predicted) %>%
#   pivot_longer(cols = c(TrueTemp, Predicted),
#                names_to = 'type',
#                values_to = 'z')
# 
# simple_train <- data_pred %>%
#   mutate(across(c(x,y), round, digits = 2)) %>%
#   group_by(x, y, type) %>%
#   summarize(z = mean(z))
# 
# simple_train %>%
#   ggplot(aes(x, y, fill = z)) +
#   geom_raster() +
#   facet_wrap(~type)
# 
# # Evaluate test RMSE
# yhat <- model %>%
#   predict(x_test)
# sqrt(mean( (y_test - yhat)^2 ))
