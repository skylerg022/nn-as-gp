##
## Spatial Basis for Spatial Categories
##

## Libraries
library(tidyverse)
# library(RSpectra)

## Read in the Data
setwd("C:/Users/skyle/Dropbox/2021 Fall/research/Quant90K")
dataset_num <- 1
filename_train <- paste0('data/dataset', dataset_num, '_training.csv')
filename_test <- paste0('data/dataset', dataset_num, '_testing.csv')
data_train <- read_csv(filename_train)
data_test <- read_csv(filename_test)

## Calculate Hughes & Haran Basis Functions
# A <- ngspatial::adjacency.matrix(size)
# P.orthog <- -(1/(size^2))*matrix(1, nrow=size^2, ncol=size^2)
# diag(P.orthog) <- 1-diag(P.orthog)
# M <- (P.orthog%*%A%*%P.orthog) %>% eigs_sym(., which="LA", k=250)
# sub.df2 <- bind_cols(sub.df, as.data.frame(M$vectors))
# ggplot(data=sub.df2, aes(x=x, y=y, fill=V250)) + 
#   geom_raster()

## Calculate Wendland basis
wendland <- function(d){
  ((1-d)^6)*(35*d^2+18*d+3)/3
}


# Create grid of bases/knots ----------------------------------------------------
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

colnames(X) <- paste0("X", 1:ncol(X))
data_train2 <- bind_cols(data_train, as.data.frame(X))
rm(X)

# Observe one of the bases
data_train2 %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(basis = mean(X21)) %>%
  ggplot(aes(x, y, fill = basis)) +
  geom_raster()

## Test the X's to see how they do
# glmtst <- glm(values ~ . - x - y, 
#               data = data_train2)
fit <- lm(values ~ . - x - y,
          data = data_train2)
predicted <- predict(fit, data_train2)

data_pred <- cbind(data_train2 %>% select(x:values), 
                   predicted) %>%
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

# data_train$fittst <- fitted(glmtst)
# p1 <- data_train %>%
#   mutate(across(c(x,y), round, digits = 2)) %>%
#   group_by(x, y) %>%
#   summarize(z = values) %>%
#   ggplot(aes(x=x, y=y, fill=z)) + 
#   geom_raster()
# p2 <- data_train %>%
#   mutate(across(c(x,y), round, digits = 2)) %>%
#   group_by(x, y) %>%
#   summarize(z = fittst) %>%
#   ggplot(aes(x=x, y=y, fill=z)) + 
#   geom_raster()
# gridExtra::grid.arrange(p1, p2)



# Fit a neural network to bases -------------------------------------------

library(keras)

x_train <- data_train2 %>%
  select(-c(x:values)) %>%
  as.matrix.data.frame()
y_train <- data_train2 %>%
  select(values) %>%
  as.matrix.data.frame()
x_test <- data_test %>%
  as.matrix.data.frame()

make_model <- function() {
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 2^8, input_shape = c(400), activation = "relu") %>%
    layer_dense(units = 2^8, activation = "relu") %>%
    layer_dense(units = 1)
  model %>%
    compile(loss = 'mse',
            optimizer = optimizer_adam())
}

set.seed(1812)
model <- make_model()

# Evaluate performance on validation
history <- model %>% 
  fit(x_train, y_train, 
      epochs = 30, batch_size = 2^7, 
      validation_split = 0.2)

# Validation MSE for 2^9 node hidden layer: about 2.75
# Validation MSE for 2^14 node hidden layer: about 2.25
# Validation MSE for 3-layer 2^9 2^8 2^7 relu: 0.35 - 0.5
beepr::beep()


# Predictions after fitting 80% of training set
predicted <- model %>%
  predict(x_train)

data_pred <- cbind(data_train2 %>% select(x:values), 
                   predicted) %>%
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