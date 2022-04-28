## Skyler Gray
## Exploratory data analysis of quant150k data

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

# Source Defaults: load libraries and important functions
source('../../functions/Defaults.R')
dirCheck()


# Read in data
load('data/AllSatelliteTemps.RData')
data_train <- all.sat.temps %>%
  rename(x = Lon, y = Lat) %>%
  filter(!is.na(MaskTemp)) %>%
  rename(values = TrueTemp)
data_test <- all.sat.temps %>%
  rename(x = Lon, y = Lat) %>%
  filter(is.na(MaskTemp) & !is.na(TrueTemp)) %>%
  rename(values = TrueTemp)
data_test2 <- all.sat.temps %>%
  rename(x = Lon, y = Lat) %>%
  filter(is.na(MaskTemp) & is.na(TrueTemp)) %>%
  rename(values = TrueTemp)
rm(all.sat.temps)


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
data_test2 %>%
  ggplot(aes(x, y)) +
  geom_bin_2d()
# Not uniform across 2d space, highly concentrated in upper right of space

# Are there multiple observations in one location?
ndistinct <- data_train %>%
  bind_rows(data_test, data_test2) %>%
  select(x, y) %>%
  distinct() %>%
  nrow()
ndistinct / ( nrow(data_train) + nrow(data_test) + nrow(data_test2) )
# All observations in the combined training and test data are distinct
#  and have a unique location

# Visualize data
# points not evenly spaced from each other. Averaging values in each "bin"
# and displaying a 100 x 100 representation of the spatial data
data_train %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(Value = mean(values)) %>%
  ggplot(aes(x, y, fill = Value)) +
  geom_raster() +
  scale_fill_continuous(type = 'viridis')

ggsave('pics/train.png',
       width = pic_width * (2/3),
       height = pic_height,
       units = pic_units,
       bg = 'white')

# Just test performance
data_test %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(Value = mean(values)) %>%
  ggplot(aes(x, y, fill = Value,
             xmin = min(x), xmax = max(x),
             ymin = min(y), ymax = max(y))) +
  geom_rect(fill = 'gray20') +
  geom_raster() +
  scale_fill_continuous(type = 'viridis')

ggsave('pics/test.png',
       width = pic_width/3*2,
       height = pic_height,
       units = pic_units,
       bg = 'white')



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
# Visualize validation set
data_train2 %>%
  mutate(across(c(x,y), round, digits = 2),
         training = 1 - validation) %>%
  group_by(x, y) %>%
  summarize(training = max(training)) %>%
  mutate(Dataset = ifelse(training == 1, 'Train', 'Val')) %>%
  ggplot(aes(x, y, fill = Dataset)) +
  geom_raster() +
  scale_fill_brewer(type = 'qual', palette = 6) +
  theme_minimal()

# Proportion of training data to be withheld for hyperparameter tuning
mean(data_train2$validation)

ggsave('pics/trainvalsplit.png',
       width = pic_width * 2/3,
       height = pic_height,
       units = pic_units,
       bg = 'white')


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

x_test2 <- data_test2[,c(1,2)] %>%
  as.matrix()
y_test2 <- data_test2[,4,drop = FALSE] %>%
  as.matrix()


# Save
save(x_train, y_train,
     x_val, y_val,
     x_test, y_test, 
     x_test2, y_test2,
     file = 'data/SatelliteTempsSplit.RData')
