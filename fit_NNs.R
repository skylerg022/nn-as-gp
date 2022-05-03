
# This code is meant to be run via the bash command line using
#  R CMD BATCH --no-save '--args <dataset name> <binary_data>' fit_NNs.R <logfile>.Rout &
#  where <dataset name> is a string for the name of the directory holding the
#  eda.R and data files of a given dataset and <binary_data> is either TRUE or
#  FALSE to indicate whether the response data is binary or continuous.
#  <logfile>.Rout is the file that will record output produced by the R file.
#  The ampersand symbol (&) at the end will make this job run in the background.
args <- commandArgs(trailingOnly = TRUE)

# ...however, uncomment the following lines to run this code in RStudio
# Set working directory if using RStudio
# if (rstudioapi::isAvailable()) {
#   setwd(dirname(rstudioapi::getSourceEditorContext()$path))
# }
# args <- c('Binary1Million', TRUE)

# Check that dataset exists
dirname <- paste0(getwd(), '/datasets/', args[1])
if (!dir.exists(dirname)) {
  message <- paste0('Directory ', dirname, ' does not exist.')
  stop(message)
}
setwd(dirname)

# If second argument is FALSE, validation loss is MSE and must 
#  be transformed to RMSE
binary_data <- as.logical(args[2])

source('../../functions/Defaults.R')
source('../../functions/Preprocess.R')
source('../../functions/NNFunctions.R')

# Read in data & model settings
load('data/DataSplit.RData')
n_train <- nrow(x_train) + nrow(x_val)

models <- read_csv('data/final_models_setup.csv')
# Only keep 4 of the 8 models for fitting:
#   The raw input types and the best non-identity
#   basis function expansion input type 
#   (transformed, radial basis, or multi-resolution basis)
final_models <- models %>%
  filter(type == 'nn') %>%
  bind_rows(models %>%
              filter(type != 'nn') %>%
              group_by(pars_type) %>%
              filter(min_loss == min(min_loss)))


# Create prediction plots in pics/final/ and return model performance metrics
n_models <- nrow(final_models)
model_metrics <- matrix(NA, nrow = n_models, ncol = 10)
for (i in 1:n_models) {
# model_metrics <- apply(final_models, 1, function(curmodel) {
  curmodel <- final_models[i,]
  # Raise epochs to nearest multiple of 5
  remainder <- curmodel$min_loss_epoch %% 5
  total_epochs <- ifelse(remainder > 0, curmodel$min_loss_epoch + (5 - remainder),
                         curmodel$min_loss_epoch)
  # Roughly decay rate per epoch
  new_decay <- ifelse(curmodel$decay_rate > 0,
                      0.1 / (n_train %/% curmodel$batch_size), 0)
  
  # Prep hyperparameters for model fitting
  if (curmodel$pars_type == 'custom') {
    model_pars <- c(n_layers = curmodel$n_layers, 
                    layer_width = curmodel$layer_width,
                    epochs = total_epochs, 
                    batch_size = curmodel$batch_size,
                    decay_rate = new_decay, 
                    dropout_rate = curmodel$dropout_rate,
                    model_num = curmodel$model_num)
  } else if (curmodel$pars_type == 'lee2018') {
    model_pars <- c(n_layers = curmodel$n_layers, 
                    layer_width = curmodel$layer_width, 
                    learning_rate = curmodel$learning_rate,
                    weight_decay = curmodel$weight_decay, 
                    epochs = total_epochs, 
                    batch_size = curmodel$batch_size, 
                    sigma_w = curmodel$sigma_w, 
                    sigma_b = curmodel$sigma_b, 
                    model_num = curmodel$model_num)
  }
  
  # Set hyperparameters for input data
  if (curmodel$type %in% c('nn', 'nn_trans')) {
    type <- curmodel$type
    sqrt_n_knots <- NULL
  } else if (curmodel$type == 'basis_4by4') {
    type <- 'basis'
    sqrt_n_knots <- c(4)
  } else if (curmodel$type == 'basis_4by4_20by20') {
    type <- 'basis'
    sqrt_n_knots <- c(4, 20)
  }
  
  # Fit NN
  fit <- evalNetwork(x_train, y_train, x_val, y_val,
                     model_pars = model_pars, 
                     x_test = x_test, y_test = y_test,
                     binary_data = binary_data,
                     type = type, 
                     pars_type = curmodel$pars_type,
                     sqrt_n_knots = sqrt_n_knots)
  
  train_mets <- unlist(fit$metrics$train)
  names(train_mets) <- paste0('train_', names(train_mets))
  test_mets <- unlist(fit$metrics$test)
  names(test_mets) <- paste0('test_', names(test_mets))
  
  mets <- c(train_mets, test_mets)
  model_metrics[i,] <- mets
}

# Give metric names to columns
colnames(model_metrics) <- names(mets)

# Add model performance to model
final_performance <- final_models %>%
  bind_cols(as.data.frame(model_metrics))

# Save final model performance metrics
write_csv(final_performance, 'data/final4_models_performance.csv')
