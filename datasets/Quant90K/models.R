

# 9-layer; 2^9, 2^8, ..., 2^1 nodes; val MSE: 0.4?
make_model <- function() {
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 2^9, input_shape = c(2), activation = "relu") %>%
    layer_dense(units = 2^8, activation = "relu") %>%
    layer_dense(units = 2^7, activation = "relu") %>%
    layer_dense(units = 2^6, activation = "relu") %>%
    layer_dense(units = 2^5, activation = "relu") %>%
    layer_dense(units = 2^4, activation = "relu") %>%
    layer_dense(units = 2^3, activation = "relu") %>%
    layer_dense(units = 2^2, activation = "relu") %>%
    layer_dense(units = 2^1, activation = "relu") %>%
    layer_dense(units = 1)
  model %>%
    compile(loss = 'mse',
            optimizer = optimizer_adam())
}


# 3-layer; 2^9 each layer; val MSE: 0.24
# Note: Predicting the whole space took about 10 seconds
make_model <- function() {
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 2^9, input_shape = c(2), activation = "relu") %>%
    layer_dense(units = 2^9, activation = "relu") %>%
    layer_dense(units = 2^9, activation = "relu") %>%
    layer_dense(units = 1)
  model %>%
    compile(loss = 'mse',
            optimizer = optimizer_adam())
}


# 2-layer; 2^9 each layer; 50% dropout each layer; val MSE: 0.52
# Note: Epochs have been 30 previously (seems to level off there),
#  but dropout this intense led to 90 epochs of training
# Results: dropout may not be a good idea, as it leads to a generalization
#  (or smoothing) of prediction values across the sample space
make_model <- function() {
  rate <- 0.5
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 2^9, input_shape = c(2), activation = "relu") %>%
    layer_dropout(rate = rate) %>%
    layer_dense(units = 2^9, activation = "relu") %>%
    layer_dropout(rate = rate) %>%
    layer_dense(units = 1)
  model %>%
    compile(loss = 'mse',
            optimizer = optimizer_adam())
}
