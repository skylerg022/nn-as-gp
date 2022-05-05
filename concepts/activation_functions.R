
# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

source('../functions/Defaults.R')

f <- function(x) (x > 0) * x
x <- seq(-5, 5, by = 0.1)
y <- f(x)

p <- ggplot(mapping = aes(x, y)) +
  geom_line(size = 1) +
  theme_minimal() +
  labs(y = 'f(x)')
p
myggsave('relu.png', plot = p)
myggsave('relu.pdf', plot = p)
