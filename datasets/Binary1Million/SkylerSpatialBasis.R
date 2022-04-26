##
## Spatial Basis for Spatial Categories
##

## Libraries
library(tidyverse)
library(raster)
library(RSpectra)

theme_set(theme_minimal())

## Read in the Data
# Set working directory if using RStudio
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}
file.name <- "data/grf_x10_y10_vf0.1_seed1.tif"
dat <- raster(file.name)

## Subset it to size x size
size <- 1001 # Size of 1001 is the full image for Sandia's images
rand.row <- sample(1:(nrow(dat)-size+1), 1)
rand.col <- sample(1:(ncol(dat)-size+1), 1)
sub.dat <- as.matrix(dat)[rand.row:(rand.row+size-1), rand.col:(rand.col+size-1)]

## Convert to dataframe and plot
sub.df <- data.frame(x=rep(1:ncol(sub.dat), nrow(sub.dat)),
                     y=rep(1:nrow(sub.dat), each=ncol(sub.dat)),
                     z=as.factor(c(sub.dat)))
ggplot(data=sub.df, aes(x=x, y=y, fill=z)) + 
  geom_raster()

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

# n.basis <- 500
# design <- maximin::maximin.cand(n.basis,
#             Xcand=sub.df[,1:2],
#             Tmax=nrow(dat))$inds

## Segment Image into 20 x 20 sections
n.parts <- 20
part.lab <- cut(sub.df$x, breaks=n.parts, labels=1:n.parts):
 cut(sub.df$y, breaks=n.parts, labels=1:n.parts)
## Sample a "knot" within each section
available <- 1:nrow(sub.df)
tst <- aggregate(available, by=list(part=part.lab[available]),
           FUN=sample, size=1)$x # one batch
qplot(sub.df$x[tst], sub.df$y[tst], geom="point")

design <- tst
# plot(sub.df[design,1], sub.df[design,2], pch=19, col="red")
theta <- 1.5*max(apply(fields::rdist(sub.df[design, c('x', 'y')], 
                                     sub.df[design, c('x', 'y')]),
                   1, function(x){
                     sort(x)[2]
                   }))
D <- fields::rdist(sub.df[, c('x', 'y')], sub.df[design, c('x', 'y')])/theta
X <- wendland(D)
X[D>1] <- 0
colnames(X) <- paste0("X", 1:ncol(X))
sub.df2 <- bind_cols(sub.df, as.data.frame(X))
ggplot(data=sub.df2, aes(x=x, y=y, fill=X17)) + 
  geom_raster()

## Test the X's to see how they do
glmtst <- glm(z~.-x-y, data=sub.df2, family="binomial")
sub.df$fittst <- fitted(glmtst)
p1 <- ggplot(data=sub.df, aes(x=x, y=y, fill=z)) + 
  geom_raster()
p2 <- ggplot(data=sub.df, aes(x=x, y=y, fill=fittst)) + 
  geom_raster()
gridExtra::grid.arrange(p1, p2)

## Write out the dataset
write.csv(x=sub.df2, file="./SkylerData400Basis.csv", row.names=FALSE)





