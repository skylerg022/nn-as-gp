library(tidyverse)
library(geoR)
library(keras)
library(tictoc)

# Temp: set working directory
setwd("C:/Users/skyle/Dropbox/2021 Fall/research/Quant90K")

# Read in data
dataset_num <- 1
data_train <- read_csv( paste0('data/dataset', dataset_num, '_training.csv') )
data_test <- read_csv( paste0('data/dataset', dataset_num, '_testing.csv') )

# Neural Network --------------------------------------------------------

x_train <- data_train[,1:2] %>%
  as.matrix.data.frame()
y_train <- data_train[,3, drop = FALSE] %>%
  as.matrix.data.frame()
x_test <- data_test %>%
  as.matrix.data.frame()

make_model_fastest <- function() {
  model <- keras_model_sequential()
  activation <- 'relu'
  model %>%
    layer_dense(units = 2^3, input_shape = c(2), activation = activation) %>%
    layer_dense(units = 1)
  model %>%
    compile(loss = 'mse',
            optimizer = optimizer_adam())
}

make_model_slowest <- function() {
  model <- keras_model_sequential()
  activation <- 'relu'
  model %>%
    layer_dense(units = 2^7, input_shape = c(2), activation = activation) %>%
    layer_dense(units = 2^7, activation = activation) %>%
    layer_dense(units = 2^7, activation = activation) %>%
    layer_dense(units = 2^7, activation = activation) %>%
    layer_dense(units = 2^7, activation = activation) %>%
    layer_dense(units = 2^7, activation = activation) %>%
    layer_dense(units = 2^7, activation = activation) %>%
    layer_dense(units = 2^7, activation = activation) %>%
    layer_dense(units = 1)
  model %>%
    compile(loss = 'mse',
            optimizer = optimizer_adam())
}

set.seed(1812)

# prediction speed of whole dataset with fastest model
tic('Fastest Model')
model <- make_model_fastest()
predicted <- model %>%
  predict(rbind(x_train, x_test))
toc()

# prediction speed of whole dataset with slowest model
tic('Slowest Model')
model <- make_model_slowest()
predicted <- model %>%
  predict(rbind(x_train, x_test))
toc()