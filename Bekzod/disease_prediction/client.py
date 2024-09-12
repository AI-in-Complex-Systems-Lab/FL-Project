import tensorflow as tf
from sklearn.model_selection import train_test_split
from dataset import load_heart_disease_data
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import ipaddress
import os
import sys
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import flwr as fl
import json
import csv

from server import save_metrics

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

with open("server_config.json", "r") as f:
    config = json.load(f)

server_addr = config["server_address"]


def get_ip_address_from_json():
    
    ip_addr = config["ip_address"]
    return ip_addr


if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description='Flower straggler / client implementation')
	parser.add_argument("-a", "--address", help="Aggregator server's IP address", default=get_ip_address_from_json())
	parser.add_argument("-p", "--port", help="Aggregator server's serving port", default=8080, type=int)
	parser.add_argument("-i", "--id", help="client ID", default=1, type=int)
	parser.add_argument("-d", "--dataset", help="dataset directory", default="/Users/guest1/Documents/GitHub/FL-Project/Bekzod/disease_prediction/Dataset")
	args = parser.parse_args()
    
try:
	ipaddress.ip_address(args.address)
except ValueError:
	sys.exit(f"Wrong IP address: {args.address}")
if args.port < 0 or args.port > 65535:
	sys.exit(f"Wrong serving port: {args.port}")
if not os.path.isdir(args.dataset):
	sys.exit(f"Wrong path to directory with datasets: {args.dataset}")

# if args.id == 3:
#       df_train = pd.read_csv(os.path.join(args.dataset, f'client_train_data_3p.csv'))
# else:
df_train = pd.read_csv(os.path.join(args.dataset, f'client_train_data_{args.id}.csv'))

df_test = pd.read_csv(os.path.join(args.dataset, 'test_data.csv'))

print(df_train.columns)
print(df_test.columns)

X_train = df_train.drop(columns=['HeartDisease']).to_numpy()
y_train = df_train['HeartDisease'].to_numpy()
X_test = df_test.drop(columns=['HeartDisease']).to_numpy()
y_test = df_test['HeartDisease'].to_numpy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Load and compile Keras model
model = keras.Sequential([
        layers.Dense(17, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),  # Dropout with a 50% drop rate (adjust as needed)
        layers.Dense(34, activation='relu'),
        Dropout(0.2),  # Dropout with a 50% drop rate (adjust as needed)
        layers.Dense(2, activation='sigmoid')
    ])

    # model.compile(
    #     loss="categorical_crossentropy",
    #     optimizer="adam",
    #     metrics=["accuracy"]
    # )

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10) 

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.metrics_dir = "metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.metrics_dir, f"client_{self.client_id}_metrics.csv")
        with open(self.metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["round", "train_loss", "train_accuracy", "eval_loss", "eval_accuracy", "f1_score"])
    
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        print("\n\n\n----------------  Train ----------------- ")
        model.set_weights(parameters)
        r = model.fit(X_train_scaled, y_train_cat, epochs=20, validation_data=(X_test_scaled, y_test_cat), verbose=0, batch_size=64, callbacks=[early_stop])
        hist = r.history
        train_loss = hist.get('loss', [None])[-1]
        train_accuracy = hist.get('accuracy', [None])[-1]
        round_number = config.get("server_round", None)
        print("Fit history: ", hist)
        print(f"Training finished for round {round_number} on client {self.client_id}")
        # Optionally save train metrics
        self.save_metrics(round_number, train_loss, train_accuracy, None, None, None)
        return model.get_weights(), len(X_train_scaled), {}

    def evaluate(self, parameters, config):
        print("\n\n\n----------------  Test ----------------- ")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
        f1 = f1_score(y_test, np.argmax(model.predict(X_test_scaled), axis=1), average='weighted')
    
        # Save metrics to client file
        round_number = config.get("server_round", None)
        self.save_metrics(round_number, None, None, loss, accuracy, f1)
    
        print(f"Evaluation results for client {self.client_id}: Loss = {loss}, Accuracy = {accuracy}, F1-Score = {f1}")
        return loss, len(X_test_scaled), {"accuracy": accuracy, "f-1 score": f1}


# Read server address from the config file

# Print the server address
print(f"Starting Flower server at {args.address}:{args.port}")
client_id = args.id

# Start Flower client
fl.client.start_client(
    server_address=f"{args.address}:{args.port}",
    client=FlowerClient(client_id),
)