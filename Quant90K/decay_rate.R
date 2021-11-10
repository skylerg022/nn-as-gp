library(ggplot2)

learningRateUpdate <- function(epoch, epoch_decay) {
  # train_size <- 72000
  # batch_size <- 2^7
  # updates_per_epoch <- train_size %/% batch_size
  learning_rate <- 0.001
  
  learning_rate * 1 / (1 + epoch_decay*(epoch-1))
}
epoch <- 1:100

qplot(epoch, learningRateUpdate(epoch, 0.5),
      geom = 'line', ylab = 'learning rate',
      ylim = c(0, NA)) +
  geom_hline(yintercept = 0, linetype = 2, col = 'red') +
  theme_bw()

