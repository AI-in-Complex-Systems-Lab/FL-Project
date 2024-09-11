import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import load_heart_disease_data
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score

# Load dataset using your mydataset.py
X, y = load_heart_disease_data()

# Use the remaining data for client 2 (ensure non-overlapping with client 1)
X_client1, X_client2, y_client1, y_client2 = train_test_split(X, y, test_size=0.5, random_state=43)

# Split client2 data into training and test sets
X_train_client2, X_test_client2, y_train_client2, y_test_client2 = train_test_split(X_client2, y_client2, test_size=0.2, random_state=43)
print(f"Client 2 training on {len(X_train_client2)} samples")

# Load and compile Keras model for client 2
model_client2 = keras.Sequential([
    layers.Dense(17, activation='relu', input_shape=(X_train_client2.shape[1],)),
    Dropout(0.2),  # Dropout layer to avoid overfitting
    layers.Dense(34, activation='relu'),
    Dropout(0.2),
    layers.Dense(2, activation='sigmoid')
])

print(f"This is the shape{X_train_client2.shape}")

model_client2.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Define Flower client for client 2
class FlowerClient2(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model_client2.get_weights()

    def fit(self, parameters, config):
        print("\n\n\n----------------  Train ----------------- ")
        model_client2.set_weights(parameters)
        r = model_client2.fit(X_train_client2, y_train_client2, epochs=5, validation_data=(X_test_client2, y_test_client2), verbose=0, batch_size=32)
        hist = r.history
        print("Fit history : ", hist)
        return model_client2.get_weights(), len(X_train_client2), {}

    def evaluate(self, parameters, config):
        print("\n\n\n----------------  Test ----------------- ")
        model_client2.set_weights(parameters)
        loss, accuracy = model_client2.evaluate(X_test_client2, y_test_client2, verbose=0)
        print("Eval accuracy : ", accuracy)
        y_pred = model_client2.predict(X_test_client2)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test_client2, axis=1)
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        print("F1 Score : ", f1)

        return loss, len(X_test_client2), {"accuracy": accuracy, "f1_score": f1}

# Start Flower client for client 2
fl.client.start_numpy_client(
    server_address="localhost:8082", 
    client=FlowerClient2(), 
    grpc_max_message_length=1024*1024*1024
)
