## Skyler Gray
## Exploratory data analysis of simulated quant150k training data

## Libraries
library(tidyverse)

# set working directory if necessary
tryCatch(setwd("C:/Users/skyle/Dropbox/2021 Fall/research/Quant150K_Sim"),
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

# EDA ---------------------------------------------------------------------

# How is the spatial data distributed?
data_train %>%
  ggplot(aes(x)) +
  geom_histogram()
data_train %>%
  ggplot(aes(y)) +
  geom_histogram()
# The data is approximately distributed uniformly across x,
#  not y (right skewed).

# Are there multiple observations in one location?
ndistinct <- data_train %>%
  bind_rows(data_test) %>%
  select(x, y) %>%
  distinct() %>%
  nrow()
ndistinct / ( nrow(data_train) + nrow(data_test) )
# All observations in the combined training and test data are distinct
#  and have a unique location

# Correlation looks apparent. Show variogram
# vario <- variog(coords = cbind(simple_train$x, simple_train$y),
#                 data = simple_train$values)
# plot(vario)
# Correlation present

# Visualize data
# points not evenly spaced from each other. Averaging values in each "bin"
# and displaying a 100 x 100 representation of the spatial data
data_train %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(Temp = mean(MaskTemp)) %>%
  ggplot(aes(x, y, fill = Temp)) +
  geom_raster() +
  scale_fill_continuous(type = 'viridis') +
  theme_minimal()


# Create localized areas to be validation set
data_train2 <- data_train %>%
  mutate(validation = ifelse( (((x > -94.5 & x < -93) & (y < 34.75)) |
                               ((x > -93.5 & x < -93) & (y > 35.5 & y < 36)) |
                               ((x > -91.75) & (y > 35.25 & y < 35.75)) |
                               ((x > -95.75 & x < -95.25) & (y > 36.25 & y < 36.75)) |
                               ((x > -95 & x < -94.5) & (y > 35.25 & y < 35.75)) |
                               ((x > -92.5 & x < -91.75) & (y > 36 & y < 36.75))),
                              1, 0))
data_train2 %>%
  mutate(across(c(x,y), round, digits = 2),
         training = 1 - validation) %>%
  group_by(x, y) %>%
  summarize(training = max(training)) %>%
  mutate(Dataset = ifelse(training == 1, 'Training', 'Validation')) %>%
  ggplot(aes(x, y, fill = Dataset)) +
  geom_raster() +
  scale_fill_brewer(type = 'qual', palette = 6)

mean(data_train2$validation)
         