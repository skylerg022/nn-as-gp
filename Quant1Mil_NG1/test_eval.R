
library(tidyverse)

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../HelperFunctions/Defaults.R')

preds <- read_csv('data/dataset2_testpreds.csv')
y <- read_csv('data/dataset2_testtrue.csv',
              col_names = FALSE) %>%
  pull()

apply(preds %>%
        select(-c(x, y)),
      2, function(yhat) {
        sqrt(mean( (yhat - y)^2 ))
      }) %>%
  round(5)


error <- preds
error[,3:10] <- apply(preds %>%
                        select(-c(x, y)),
                      2, function(yhat) {
                        yhat - y
                      })

# Post-hoc analysis
simple_test <- error %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  pivot_longer(latlong_nn_custom:basis_4by4_20by20_lee,
               names_to = 'model', 
               values_to = 'yhat') %>%
  group_by(x, y, model) %>%
  summarize(yhat = mean(yhat))

# Just test performance
p_test <- simple_test %>%
  filter(model == 'basis_4by4_20by20_custom') %>%
  ggplot(aes(x, y, fill = yhat)) +
  geom_raster() +
  # facet_wrap(~model) +
  scale_fill_gradient2() +
  theme_minimal() +
  labs(fill = 'Value')

p_test %>%
  ggsave('pics/error_4by4_20by20_custom.png', plot = .,
         width = pic_width/3*2,
         height = pic_height,
         units = pic_units,
         bg = 'white')
