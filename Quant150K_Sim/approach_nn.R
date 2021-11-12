library(tidyverse)
#library(geoR)
library(keras)

# set working directory if necessary
tryCatch(setwd('C:/Users/skyle/Documents/GithubRepos/nn-as-gp/Quant150K_Sim'),
         error = function(cond) {
           message(paste0('Could not set directory. ',
                          'Assuming code is being run via Bash.'))
         })
source('../HelperFunctions/MakeNNModel.R')

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

# Scaling variables
n_80perc <- floor(nrow(data_train) * 0.8)
mean_80perc <- data_train[,1:2] %>%
  head(n_80perc) %>%
  colMeans()
sd_80perc <- data_train[,1:2] %>%
  head(n_80perc) %>%
  apply(2, sd)

n <- nrow(data_train)
mean_all <- data_train[,1:2] %>%
  colMeans()
sd_all <- data_train[,1:2] %>%
  apply(2, sd)

quickMatrix <- function(n, x) {
  matrix(x, ncol = length(x), nrow = n, byrow = TRUE)
}

# Neural Network --------------------------------------------------------

# Train on 80% of the training data and validate
x_train <- ( (data_train[,1:2] - quickMatrix(n, mean_80perc)) /
               quickMatrix(n, sd_80perc) ) %>%
  as.matrix()
y_train <- data_train[,3, drop = FALSE] %>%
  as.matrix.data.frame()

model_pars <- c(n_layers = 6, layer_width = 2^7,
                epochs = 20, batch_size = 2^7,
                decay_rate = 0, square_feat = 0,
                dropout_rate = 0.1, model_num = 1)

model <- fitModel(model_pars, cbind(x_train, x_train^2), y_train, test = 'part_train')

# What areas are solely validation data?
# data_train %>%
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
  predict(cbind(x_train, x_train^2))

data_pred <- cbind(data_train, Predicted) %>%
  # clip predictions to max/min of observed data
  # mutate(Predicted = ifelse(Predicted > max(TrueTemp), max(TrueTemp), Predicted),
  #        Predicted = ifelse(Predicted < min(TrueTemp), min(TrueTemp), Predicted)) %>%
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

x_train <- ( (data_train[,1:2] - quickMatrix(n, mean_all)) /
               quickMatrix(n, sd_all) ) %>%
  as.matrix()
y_train <- data_train[,3, drop = FALSE] %>%
  as.matrix()
x_test <- ( (data_test[,1:2] - quickMatrix(nrow(data_test), mean_all)) /
              quickMatrix(nrow(data_test), sd_all) ) %>%
  as.matrix()
y_test <- data_test[,4, drop = FALSE] %>%
  as.matrix.data.frame()

model <- fitModel(model_pars, cbind(x_train, x_train^2), y_train, test = 'all_train')

# Predictions (including test set) after fitting 100% of training set
Predicted <- model %>%
  predict(rbind(cbind(x_train, x_train^2), cbind(x_test, x_test^2)))

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
  geom_raster(data = data_train,
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
  predict(cbind(x_test, x_test^2))
sqrt(mean( (y_test - yhat)^2 ))

# Evaluate train RMSE
yhat <- model %>%
  predict(cbind(x_train, x_train^2))
sqrt(mean( (y_train - yhat)^2 ))
