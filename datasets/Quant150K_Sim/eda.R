## Exploratory data analysis of quant150k simulated training data

## Libraries
library(tidyverse)

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

# Read in data
load('data/AllSimulatedTemps.RData')
data_train <- all.sim.data %>%
  rename(x = Lon, y = Lat) %>%
  filter(!is.na(MaskTemp)) %>%
  rename(values = MaskTemp)
data_test <- all.sim.data %>%
  rename(x = Lon, y = Lat) %>%
  filter(is.na(MaskTemp)) %>%
  rename(values = MaskTemp)
rm(all.sim.data)


# EDA ---------------------------------------------------------------------

# How is the spatial data distributed?
data_train %>%
  ggplot(aes(x, y)) +
  geom_bin_2d()
# The data is not uniform across 2d space

# How about for test data?
data_test %>%
  ggplot(aes(x, y)) +
  geom_bin_2d()
# Not uniform across 2d space, highly concentrated in upper right of space

# Are there multiple observations in one location?
ndistinct <- data_train %>%
  bind_rows(data_test) %>%
  select(x, y) %>%
  distinct() %>%
  nrow()
ndistinct / ( nrow(data_train) + nrow(data_test) )
# All observations in the combined training and test data are distinct
#  and have a unique location

# Visualize data
# points not evenly spaced from each other. Averaging values in each "bin"
# and displaying a 100 x 100 representation of the spatial data
p <- data_train %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(Value = mean(values)) %>%
  ggplot(aes(x, y, fill = Value)) +
  geom_raster() +
  scale_fill_continuous(type = 'viridis')
p
myggsave(filename = 'pics/train.png', plot = p)
myggsave(filename = 'pics/train.pdf', plot = p)

# Just test performance
p <- data_test %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(Value = mean(TrueTemp)) %>%
  ggplot(aes(x, y, fill = Value)) +
  geom_raster() +
  scale_fill_continuous(type = 'viridis')
p
myggsave(filename = 'pics/test.png', plot = p)
myggsave(filename = 'pics/test.pdf', plot = p)


# Choosing a validation set -----------------------------------------------

# Create localized areas to be validation set
data_train2 <- data_train %>%
  mutate(validation = ifelse( (((x > -94.5 & x < -93) & (y < 34.75)) |
                                 ((x > -93.5 & x < -93) & (y > 35.5 & y < 36)) |
                                 ((x > -91.75) & (y > 35.25 & y < 35.75)) |
                                 ((x > -95.75 & x < -95.25) & (y > 36.25 & y < 36.75)) |
                                 ((x > -95 & x < -94.5) & (y > 35.25 & y < 35.75)) |
                                 ((x > -92.5 & x < -91.75) & (y > 36 & y < 36.75))),
                              1, 0))
p <- data_train2 %>%
  mutate(across(c(x,y), round, digits = 2),
         training = 1 - validation) %>%
  group_by(x, y) %>%
  summarize(training = max(training)) %>%
  mutate(Dataset = ifelse(training == 1, 'Train', 'Val')) %>%
  ggplot(aes(x, y, fill = Dataset)) +
  geom_raster() +
  scale_fill_discrete_qualitative()
p
myggsave(filename = 'pics/trainvalsplit.png', plot = p)
myggsave(filename = 'pics/trainvalsplit.pdf', plot = p)

mean(data_train2$validation)


# Saving data -------------------------------------------------------------

# Train
train <- data_train2 %>%
  filter(validation == 0)
x_train <- train[,c(1,2)] %>%
  as.matrix()
y_train <- train[,4,drop = FALSE] %>%
  as.matrix()

# Validation
val <- data_train2 %>%
  filter(validation == 1)
x_val <- val[,c(1,2)] %>%
  as.matrix()
y_val <- val[,4,drop = FALSE] %>%
  as.matrix()

# Test
x_test <- data_test[,c(1,2)] %>%
  as.matrix()
y_test <- data_test[,4,drop = FALSE] %>%
  as.matrix()

# Save
save(x_train, y_train,
     x_val, y_val,
     x_test, y_test, 
     file = 'data/SimulatedTempsSplit.RData')
