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
import flwr as fl
import json
import csv

# Define the base directory as the current directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to ids_dnp3 directory
ids_dnp3_federated_datasets_path = os.path.join(base_dir, 'datasets', 'federated_datasets')

# Read server address from the config file
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
	parser.add_argument("-i", "--id", help="client ID", default=1, required=True, type=int)
	parser.add_argument("-d", "--dataset", help="dataset directory", default="./datasets/federated_datasets/")
	args = parser.parse_args()

	try:
		ipaddress.ip_address(args.address)
	except ValueError:
		sys.exit(f"Wrong IP address: {args.address}")
	if args.port < 0 or args.port > 65535:
		sys.exit(f"Wrong serving port: {args.port}")
	if not os.path.isdir(args.dataset):
		sys.exit(f"Wrong path to directory with datasets: {args.dataset}")

	# Make TensorFlow log less verbose
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

	# Load train and test data
	df_train = pd.read_csv(os.path.join(args.dataset, f'client_train_data_{args.id}.csv'))
	df_test = pd.read_csv(os.path.join(args.dataset, 'test_data.csv'))

	# Split data into X and y
	X_train = df_train.drop(columns=['y']).to_numpy()
	y_train = df_train['y'].to_numpy()
	X_test = df_test.drop(columns=['y']).to_numpy()
	y_test = df_test['y'].to_numpy()

	# Scale feature values for input data normalization
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# Use one-hot-vectors for label representation
	y_train_cat = to_categorical(y_train)
	y_test_cat = to_categorical(y_test)

	# Define a MLP model
	model = Sequential([
		InputLayer(input_shape=(X_train_scaled.shape[1],)),
		Dense(units=50, activation='relu'),
		Dropout(0.2),
		Dense(units=y_train_cat.shape[1], activation='softmax')
	])

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)  

class Client(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id

        # Define the directory to save metrics
        self.metrics_dir = "metrics"

        # Create the directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Define the CSV file path inside the metrics folder
        self.metrics_file = os.path.join(self.metrics_dir, f"client_{self.client_id}_metrics.csv")

        # Initialize the CSV file with headers
        with open(self.metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["round", "train_loss", "train_accuracy", "eval_loss", "eval_accuracy", "f1_score"])

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(
            X_train_scaled, y_train_cat,
            epochs=1,  # Set to 1 to fit one round per call
            validation_data=(X_test_scaled, y_test_cat),
            batch_size=64,
            callbacks=[early_stop],
            verbose=0
        )

        # Get the loss and accuracy from the training history
        train_loss = history.history.get('loss', [None])[0]
        train_accuracy = history.history.get('accuracy', [None])[0]

        # Log training metrics
        round_number = config.get("server_round", None)
        print(f"Training finished for round {round_number} on client {self.client_id}")

        # Evaluate after training
        try:
            loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
            predictions = model.predict(X_test_scaled)
            f1 = f1_score(y_test, np.argmax(predictions, axis=1), average='weighted')
            print(f"Evaluated: Loss = {loss}, Accuracy = {accuracy}, F1-Score = {f1}")

            # Check if training metrics are not None before logging
            if train_loss is not None and train_accuracy is not None:
                print(f"Logging metrics for round {round_number} on client {self.client_id}")
                with open(self.metrics_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([round_number, train_loss, train_accuracy, loss, accuracy, f1])
                print(f"Metrics successfully logged for client {self.client_id} at round {round_number}")
            else:
                print(f"Missing training metrics for client {self.client_id}, skipping log.")
        except Exception as e:
            print(f"Error during evaluation or logging for client {self.client_id} at round {round_number}: {e}")

        return model.get_weights(), len(X_train_scaled), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
        f1 = f1_score(y_test, np.argmax(model.predict(X_test_scaled), axis=1), average='weighted')
        print(f"Evaluation results for client {self.client_id}: Loss = {loss}, Accuracy = {accuracy}, F1-Score = {f1}")
        return loss, len(X_test_scaled), {"accuracy": accuracy, "f1-score": f1}


# Start Flower client and initiate communication with the Flower aggregation server
print(f"Connecting to server at {args.address}:{args.port}")

client_id = args.id
# Start Flower straggler and initiate communication with the Flower aggretation server
fl.client.start_numpy_client(
	server_address=f"{args.address}:{args.port}",
	client=Client(client_id)
) 
