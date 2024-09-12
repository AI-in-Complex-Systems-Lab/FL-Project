import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import load_heart_disease_data
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score

import json
import socket

# Load dataset using your mydataset.py
X, y = load_heart_disease_data()

# Split dataset into two non-overlapping halves for client 1 and client 2
X_client1, X_client2, y_client1, y_client2 = train_test_split(X, y, test_size=0.5, random_state=43)

# Split client1 data into training and test sets
X_train_client1, X_test_client1, y_train_client1, y_test_client1 = train_test_split(X_client1, y_client1, test_size=0.2, random_state=43)
# print(f"Client 1 training on {len(X_train_client1)} samples")

# Load and compile Keras model for client 1
model_client1 = keras.Sequential([
    layers.Dense(17, activation='relu', input_shape=(X_train_client1.shape[1],)),
    Dropout(0.2),  # Dropout layer to avoid overfitting
    layers.Dense(34, activation='relu'),
    Dropout(0.2),
    layers.Dense(2, activation='sigmoid')
])

model_client1.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Define Flower client for client 1
class FlowerClient1(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model_client1.get_weights()

    def fit(self, parameters, config):
        print("\n\n\n----------------  Train ----------------- ")
        model_client1.set_weights(parameters)
        r = model_client1.fit(X_train_client1, y_train_client1, epochs=5, validation_data=(X_test_client1, y_test_client1), verbose=0, batch_size=32)
        hist = r.history
        print("Fit history : ", hist)
        return model_client1.get_weights(), len(X_train_client1), {}
        
    def evaluate(self, parameters, config):
        print("\n\n\n----------------  Test ----------------- ")
        model_client1.set_weights(parameters)
        loss, accuracy = model_client1.evaluate(X_test_client1, y_test_client1, verbose=0)
        print("Eval accuracy : ", accuracy)
        y_pred = model_client1.predict(X_test_client1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test_client1, axis=1)
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        print("F1 Score : ", f1)

        return loss, len(X_test_client1), {"accuracy": accuracy, "f1_score": f1}


with open("server_config.json", "r") as f:
    config = json.load(f)


import pickle
import os
from sklearn.linear_model import LogisticRegression  # Example model
from sklearn.metrics import accuracy_score

# Initialize model (or load previous model state if available)
model = LogisticRegression()

# Train using saved batches
batch_files = sorted([f for f in os.listdir() if f.startswith('batch_')])
for batch_file in batch_files:
    # Load each batch without loading the entire dataset
    with open(batch_file, 'rb') as f:
        X_batch, y_batch = pickle.load(f)

    # Train the model on the batch
    model.fit(X_batch, y_batch)
    print(f"Trained on {batch_file}")

# Save the trained model for future use
with open("trained_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Training complete and model saved.")


import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('169.226.53.85', 8080))
response = client_socket.recv(1024)
print(response.decode('utf-8'))
client_socket.close()


server_addr = config["server_address"]
#print(f'Starting Flower server at {server_addr}')

# Start Flower client for client 1
fl.client.start_client(
    server_address=server_addr, 
    client=FlowerClient1(), 
)
    