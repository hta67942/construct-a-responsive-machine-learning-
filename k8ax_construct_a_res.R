# Load necessary libraries
library(tensorflow)
library(keras)
library(caret)
library(DT)

# Set seed for reproducibility
set.seed(123)

# Define a function to load and preprocess data
load_data <- function(file_path) {
  # Load data
  data <- read.csv(file_path)
  
  # Preprocess data
  data$scaled <- scale(data[, -ncol(data)])
  
  return(data)
}

# Define a function to construct and train a machine learning model
construct_model <- function(data, model_type) {
  # Construct model
  if(model_type == "neural_network") {
    model <- keras_model_sequential() %>%
      layer_dense(units = 64, activation = "relu", input_shape = ncol(data$scaled) - 1) %>%
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = 1)
    
    model %>% 
      compile(
        optimizer = optimizer_adam(),
        loss = "mean_squared_error"
      )
  } else if(model_type == "decision_tree") {
    model <- rpart(response ~ ., data = data)
  } else {
    stop("Invalid model type. Supported types are 'neural_network' and 'decision_tree'.")
  }
  
  # Train model
  if(model_type == "neural_network") {
    model %>% fit(
      x = as.matrix(data$scaled[, -ncol(data$scaled)]),
      y = data$scaled[, ncol(data$scaled)],
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2
    )
  } else if(model_type == "decision_tree") {
    model
  }
  
  return(model)
}

# Define a function to make predictions
make_predictions <- function(model, data, model_type) {
  if(model_type == "neural_network") {
    predictions <- predict(model, as.matrix(data$scaled[, -ncol(data$scaled)]))
  } else if(model_type == "decision_tree") {
    predictions <- predict(model, data, type = "response")
  }
  
  return(predictions)
}

# Define a function to evaluate model performance
evaluate_model <- function(model, data, model_type) {
  predictions <- make_predictions(model, data, model_type)
  
  if(model_type == "neural_network") {
    mse <- mean((data$scaled[, ncol(data$scaled)] - predictions)^2
  } else if(model_type == "decision_tree") {
    mse <- mean((data$response - predictions)^2)
  }
  
  return(mse)
}

# Main function
main <- function(file_path, model_type) {
  # Load and preprocess data
  data <- load_data(file_path)
  
  # Construct and train model
  model <- construct_model(data, model_type)
  
  # Evaluate model performance
  mse <- evaluate_model(model, data, model_type)
  
  print(paste("Model type:", model_type, "| MSE:", mse))
}

# Call main function
main("data.csv", "neural_network")