import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap

# Load preprocessed data
data = pd.read_csv('preprocessed_weather_data.csv')

# Define features and target variables
# Using several weather-related identifiers as input features (X) and the target (y) is the 'value' column.
X = data[['identifier_P1H', 'identifier_P24H', 'identifier_P6H', 'identifier_PMSL', 
          'identifier_RH', 'identifier_T', 'identifier_TDP', 'identifier_UVI', 'identifier_WD', 
          'identifier_WS','identifier_pressureMeanSeaLevel','location_id_23.0','location_id_27.0','location_id_30.0','location_id_116.0','location_id_152.0']].values
y = data['value'].values

# Normalize the features using MinMaxScaler
# MinMaxScaler scales the features to a range of [0, 1] for better neural network performance.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Get model parameters from user input
neurons = int(input("Enter the number of neurons in the layers (e.g. 64): "))
activation_function = input("Choose an activation function (sigmoid or relu): ")
cost_function = input("Choose a cost function (mean_squared_error or mean_absolute_error): ")
optimizer = input("Choose an optimizer (adam or sgd): ")

# Build a sequential neural network model
# The model consists of three layers: input layer, hidden layer, and output layer.
model = keras.Sequential([
    layers.Dense(neurons, activation=activation_function, input_shape=(X_train.shape[1],)),
    layers.Dense(neurons // 2, activation=activation_function),
    layers.Dense(1)  # Output layer (for regression)
])

# Compile the model with chosen optimizer and cost function (loss function)
model.compile(optimizer=optimizer, loss=cost_function)

# Train the model with 10 epochs and batch size of 32
# We also use 20% of the training data as validation data during training.
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data and print the loss
# The loss is calculated using the selected cost function
loss = model.evaluate(X_test, y_test)

# Store and display the model evaluation results
evaluation_results = {
    'neurons': neurons,
    'activation_function': activation_function,
    'cost_function': cost_function,
    'optimizer': optimizer,
    'model_loss': loss
}

print("\nModel evaluation results:")
for key, value in evaluation_results.items():
    print(f"{key}: {value}")

# Make predictions using the trained model
predictions = model.predict(X_test)

# Display the predictions
print("\nPredictions:\n", predictions)

# SHAP is used to explain the output of the model
explainer = shap.Explainer(model, X_train)

# Calculate SHAP values for the test data
shap_values = explainer(X_test)

# Display names
new_feature_names = ['P1H', 'P24H', 'P6H', 'PMSL', 'RH', 'T', 'TDP', 'UVI', 'WD', 'WS','location_id_23.0','location_id_27.0','location_id_30.0','location_id_116.0','location_id_152.0']

# Update the feature names in the SHAP values object
shap_values.feature_names = new_feature_names

# Generate a summary plot with the new feature names
shap.summary_plot(shap_values, X_test)

