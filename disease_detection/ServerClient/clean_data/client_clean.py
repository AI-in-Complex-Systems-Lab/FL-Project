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
X_clean, y_clean = load_heart_disease_data()

X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)


# Load and compile Keras model for client 1
model_client1 = tf.keras.Sequential([
    layers.Dense(17, activation='relu', input_shape=(X_train_clean.shape[1],)),
    Dropout(0.2),  # Dropout layer to avoid overfitting
    layers.Dense(34, activation='relu'),
    Dropout(0.2),
    layers.Dense(2, activation='sigmoid')
])

print(X_train_clean.shape)

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
        r = model_client1.fit(X_train_clean, y_train_clean, epochs=5, validation_data=(X_test_clean, y_test_clean), verbose=0, batch_size=32)
        hist = r.history
        print("Fit history : ", hist)
        return model_client1.get_weights(), len(X_train_clean), {}

    def evaluate(self, parameters, config):
        print("\n\n\n----------------  Test ----------------- ")
        model_client1.set_weights(parameters)
        loss, accuracy = model_client1.evaluate(X_test_clean, y_test_clean, verbose=0)
        print("Eval accuracy : ", accuracy)
        y_pred = model_client1.predict(X_test_clean)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test_clean, axis=1)
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        print("F1 Score : ", f1)

        return loss, len(X_test_clean), {"accuracy": accuracy, "f1_score": f1}

# Start Flower client for client 1
fl.client.start_numpy_client(
    server_address="localhost:8082", 
    client=FlowerClient1(), 
    grpc_max_message_length=1024*1024*1024
)
