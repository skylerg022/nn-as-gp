# Parallel processing using sockets (not forking)
## NOTE: Using sockets is slower than forking
##  but necessary for parallel processing in R on Windows

# Process that is "embarassingly parallel" (it's also redundant)
f <- function(i) {
  lmer(Petal.Width ~ . - Species + (1 | Species), data = iris)
}

library(lme4)
system.time(save1 <- lapply(1:100, f))


library(parallel)

system.time({
  numCores <- detectCores(logical = FALSE) # I have 4 physical cores, 8 logical cores
  cl <- makeCluster(numCores) # initialize cores
  clusterEvalQ(cl, library(lme4)) # load library in cores
  
  save3 <- parLapply(cl, 1:100, f)
  stopCluster(cl)
  })
