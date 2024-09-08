# client 1

import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import load_heart_disease_data
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset using mydataset.py
X, y = load_heart_disease_data()

# Split data into training and testing sets with a random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=43)

# Load and compile Keras model
model = keras.Sequential([
    layers.Dense(17, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),  # Dropout with a 50% drop rate (adjust as needed)
    layers.Dense(34, activation='relu'),
    Dropout(0.2),  # Dropout with a 50% drop rate (adjust as needed)
    layers.Dense(2, activation='sigmoid')
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        print("\n\n\n----------------  Train ----------------- ")
        model.set_weights(parameters)
        r = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=0, batch_size = 32)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        print("\n\n\n----------------  Test ----------------- ")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(X_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:8082", 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)