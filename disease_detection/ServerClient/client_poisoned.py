import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from dataset2 import load_heart_disease_data
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score

# Load dataset using your mydataset.py
X, y = load_heart_disease_data()

# Split poisoned data for training and testing
X_train_poisoned, X_test_poisoned, y_train_poisoned, y_test_poisoned = train_test_split(X, y, test_size=0.2, random_state=42)

# Remove the extra feature from poisoned data (truncate to 50 features)
if X_train_poisoned.shape[1] > 50:  # If poisoned data has 51 features
    X_train_poisoned = X_train_poisoned[:, :50]
    X_test_poisoned = X_test_poisoned[:, :50]


# Now define and compile Keras model for Client 2 with dynamic input shape
model_client2 = tf.keras.Sequential([
    layers.Dense(17, activation='relu', input_shape=(X_train_poisoned.shape[1],)),  # Adjusted input shape
    Dropout(0.2),  
    layers.Dense(34, activation='relu'),
    Dropout(0.2),
    layers.Dense(2, activation='sigmoid')
])

print(X_train_poisoned.shape)

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
        r = model_client2.fit(X_train_poisoned, y_train_poisoned, epochs=5, validation_data=(X_test_poisoned, y_test_poisoned), verbose=0, batch_size=32)
        hist = r.history
        print("Fit history : ", hist)
        return model_client2.get_weights(), len(X_train_poisoned), {}

    def evaluate(self, parameters, config):
        print("\n\n\n----------------  Test ----------------- ")
        model_client2.set_weights(parameters)
        loss, accuracy = model_client2.evaluate(X_test_poisoned, y_test_poisoned, verbose=0)
        print("Eval accuracy : ", accuracy)
        y_pred = model_client2.predict(X_test_poisoned)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test_poisoned, axis=1)
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        print("F1 Score : ", f1)

        return loss, len(X_test_poisoned), {"accuracy": accuracy, "f1_score": f1}

# Start Flower client for client 2
fl.client.start_numpy_client(
    server_address="localhost:8082", 
    client=FlowerClient2(), 
    grpc_max_message_length=1024*1024*1024
)
