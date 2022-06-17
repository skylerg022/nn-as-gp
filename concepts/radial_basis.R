# Provide visualization for basis function expansions

# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../functions/defaults.R')
source('../functions/Preprocess.R')

load('../datasets/Quant150K_Sim/data/DataSplit.RData')

x_bases <- multiResBases(x_train = rbind(x_train, x_val), x_withheld = x_test,
                         sqrt_n_knots = c(4), local_n = 30, 
                         closest_minval = 0.75, test = TRUE)

train <- cbind(rbind(x_train, x_val), x_bases$x_train) %>%
  as.data.frame()

# Observe one of the bases
p <- train %>%
  mutate(across(c(x,y), round, digits = 2)) %>%
  group_by(x, y) %>%
  summarize(basis = mean(X4_6)) %>%
  ggplot(aes(x, y, fill = basis)) +
  geom_raster() +
  labs(fill = 'Basis6')
p
myggsave('basis_example.png', plot = p)
myggsave('basis_example.pdf', plot = p)
