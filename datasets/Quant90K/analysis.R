library(tidyverse)
library(geoR)
library(keras)

# Temp: set working directory
setwd("C:/Users/skyle/Dropbox/2021 Fall/research/Quant90K")

# Read in data
dataset_num <- 1
data_train <- read_csv( paste0('data/dataset', dataset_num, '_training.csv') )
data_test <- read_csv( paste0('data/dataset', dataset_num, '_testing.csv') )



# EDA ---------------------------------------------------------------------

# How is the spatial data distributed?
data_train %>%
  ggplot(aes(x = x)) +
  geom_histogram()
data_train %>%
  ggplot(aes(x = y)) +
  geom_histogram()
# The data is distributed uniformly across the x's and y's

# Are there multiple observations in one location?
ndistinct <- data_train %>%
  select(x, y) %>%
  bind_rows(data_test) %>%
  distinct() %>%
  nrow()
ndistinct / ( nrow(data_train) + nrow(data_test) )
# No. All observations in the combined training and test data are distinct

# Plot the training spatial data
# points not evenly spaced from each other. Averaging values in each "bin"
# and displaying a 100 x 100 representation of the spatial data
simple_train <- data_train %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(values = mean(values))
  
simple_train %>%
  ggplot(aes(x, y, fill = values)) +
  geom_raster()

# Correlation looks apparent. Show variogram
vario <- variog(coords = cbind(simple_train$x, simple_train$y),
                data = simple_train$values)
plot(vario)
# Strong spatial correlation present until distance of around 0.2 or so.

data_train %>%
  mutate(across(c(x,y), round, digits = 2),
         training = ifelse(row_number()/n() < 0.8, 1, 0)) %>%
  group_by(x, y) %>%
  summarize(training = max(training)) %>%
  mutate(Dataset = ifelse(training == 1, 'Training', 'Validation')) %>%
  ggplot(aes(x, y, fill = Dataset)) +
  geom_raster() +
  scale_fill_brewer(type = 'qual', palette = 6)

# Neural Network --------------------------------------------------------

x_train <- data_train[,1:2] %>%
  as.matrix.data.frame()
y_train <- data_train[,3, drop = FALSE] %>%
  as.matrix.data.frame()
x_test <- data_test %>%
  as.matrix.data.frame()

make_model <- function() {
  model <- keras_model_sequential()
  activation <- 'relu'
  layer_width <- 2^7 #8
  n_layers <- 8 #16
  
  addLayers <- function(model, n_layers) {
    if (n_layers > 0) {
      addLayers(model, n_layers-1) %>%
        layer_dense(units = layer_width, activation = activation)
    } else model
  }
  
  model %>%
    layer_dense(units = layer_width, input_shape = c(2), activation = activation) %>%
    addLayers(n_layers - 1) %>%
    layer_dense(units = 1)
  model %>%
    compile(loss = 'mse',
            optimizer = optimizer_adam(learning_rate = 0.001,
                                       decay = 0)) #decay = 0.05/563/4))
}

set.seed(1812)
model <- make_model()

# Evaluate performance on validation
history <- model %>% 
  fit(x_train, y_train, 
      epochs = 400, batch_size = 2^7, 
      validation_split = 0.2)

# Validation MSE for 2^9 node hidden layer: about 2.75
# Validation MSE for 2^14 node hidden layer: about 2.25
# Validation MSE for 3-layer 2^9 2^8 2^7 relu: 0.35 - 0.5
# Validation MSE for 4-layer 2^8 relu: 0.23; prediction: 3 sec
# Validation MSE for 4-layer 2^8 relu, batch size 2^6, not 2^7: 0.23; prediction: 3 sec
beepr::beep()

overfit_dat <- tibble(epoch = 1:400,
                      mse = history$metrics$loss,
                      val_mse = history$metrics$val_loss)
overfit_dat %>%
  pivot_longer(cols = c(mse, val_mse),
               names_to = 'metric',
               values_to = 'loss') %>%
  filter(epoch > 100) %>%
  ggplot(aes(x = epoch, y = loss, col = metric)) +
  geom_line(size = 1) +
  theme_bw() +
  scale_color_brewer(type = 'qual', palette = 2)

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

# data_pred %>%
#   ggplot(aes(x, y, col = z)) +
#   geom_point() +
#   facet_wrap(~type, nrow = 1)

# Refit on all the data
model <- make_model()
history <- model %>% 
  fit(x_train, y_train, 
      epochs = 30, batch_size = 128)

nn_pred <- model %>%
  predict(x_train)

  
