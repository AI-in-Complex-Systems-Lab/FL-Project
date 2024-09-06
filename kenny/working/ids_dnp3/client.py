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
from tensorflow.keras.callbacks import EarlyStopping, Callback
import flwr as fl
from colorama import Fore, Style
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

# Custom callback to print and log loss and accuracy every 5 epochs
class PrintMetrics(Callback):
    def __init__(self, node_id, loss_writer, accuracy_writer, round_number):
        super().__init__()
        self.node_id = node_id
        self.loss_writer = loss_writer
        self.accuracy_writer = accuracy_writer
        self.round_number = round_number
        self.current_epoch = 0  # To track the current epoch

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch + 1  # Track the latest epoch
        # Log training loss and accuracy every 5 epochs
        if self.current_epoch % 5 == 0:
            print(
                f"Epoch {self.current_epoch}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}, "
                f"Val Loss = {logs['val_loss']:.4f}, Val Accuracy = {logs['val_accuracy']:.4f}"
            )
            # Log the metrics to the corresponding CSV files
            self.loss_writer.writerow(['train_loss', logs['loss'], self.current_epoch, self.round_number])
            self.accuracy_writer.writerow(['train_accuracy', logs['accuracy'], self.current_epoch, self.round_number])
            self.loss_writer.writerow(['test_loss', logs['val_loss'], self.current_epoch, self.round_number])
            self.accuracy_writer.writerow(['test_accuracy', logs['val_accuracy'], self.current_epoch, self.round_number])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flower straggler / client implementation')
    parser.add_argument("-a", "--address", help="Aggregator server's IP address", default=get_ip_address_from_json())
    parser.add_argument("-p", "--port", help="Aggregator server's serving port", default=8080, type=int)
    parser.add_argument("-i", "--id", help="Client ID", default=1, type=int)
    parser.add_argument("-n", "--node-id", help="Node ID for labeling", required=True, type=int)
    parser.add_argument("-d", "--dataset", help="Dataset directory", default=ids_dnp3_federated_datasets_path)
    args = parser.parse_args()

    try:
        ipaddress.ip_address(args.address)
    except ValueError:
        sys.exit(f"{Fore.RED}Wrong IP address: {args.address}{Style.RESET_ALL}")
    if args.port < 0 or args.port > 65535:
        sys.exit(f"{Fore.RED}Wrong serving port: {args.port}{Style.RESET_ALL}")
    if not os.path.isdir(args.dataset):
        sys.exit(f"{Fore.RED}Wrong path to directory with datasets: {args.dataset}{Style.RESET_ALL}")

    # Make TensorFlow log less verbose
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load train and test data
    print(f"{Fore.CYAN}Client {args.node_id} loading train and test data...{Style.RESET_ALL}")
    df_train = pd.read_csv(os.path.join(args.dataset, f'client_train_data_{args.id}.csv'))
    df_test = pd.read_csv(os.path.join(args.dataset, 'test_data.csv'))
    print(f"{Fore.CYAN}Client {args.node_id} data loaded.{Style.RESET_ALL}")

    # Split data into X and y
    X_train = df_train.drop(columns=['y']).to_numpy()
    y_train = df_train['y'].to_numpy()
    X_test = df_test.drop(columns=['y']).to_numpy()
    y_test = df_test['y'].to_numpy()

    # Scale feature values for input data normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"{Fore.CYAN}Client {args.node_id} data scaled.{Style.RESET_ALL}")

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
    print(f"{Fore.CYAN}Client {args.node_id} model compiled.{Style.RESET_ALL}")

    # Create a folder for storing metrics if it doesn't exist
    metrics_dir = os.path.join(base_dir, 'Metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    # Open CSV files for logging metrics
    loss_csv_path = os.path.join(metrics_dir, f'node_{args.node_id}_loss.csv')
    accuracy_csv_path = os.path.join(metrics_dir, f'node_{args.node_id}_accuracy.csv')
    f1_csv_path = os.path.join(metrics_dir, f'node_{args.node_id}_f1.csv')

    loss_file = open(loss_csv_path, mode='w', newline='')
    accuracy_file = open(accuracy_csv_path, mode='w', newline='')
    f1_file = open(f1_csv_path, mode='w', newline='')

    loss_writer = csv.writer(loss_file)
    accuracy_writer = csv.writer(accuracy_file)
    f1_writer = csv.writer(f1_file)

    # Write headers
    loss_writer.writerow(['metric', 'value', 'epoch', 'round'])
    accuracy_writer.writerow(['metric', 'value', 'epoch', 'round'])
    f1_writer.writerow(['metric', 'value', 'epoch', 'round'])

    # Define Flower client
    class Client(fl.client.NumPyClient):
        def __init__(self, round_number):
            self.current_epoch = 0  # Track the latest training epoch
            self.round_number = round_number

        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            print(f"{Fore.CYAN}Client {args.node_id} starting training for round {config['server_round']}...{Style.RESET_ALL}")
            history = model.fit(
                X_train_scaled, y_train_cat,
                epochs=100,
                validation_data=(X_test_scaled, y_test_cat),
                batch_size=64,
                callbacks=[early_stop, PrintMetrics(args.node_id, loss_writer, accuracy_writer, self.round_number)],
                verbose=0
            )
            self.current_epoch = len(history.history['loss'])  # Store the last epoch count from training
            print(f"{Fore.CYAN}Client {args.node_id} training finished for round {config['server_round']}{Style.RESET_ALL}")
            return model.get_weights(), len(X_train_scaled), {}

        def evaluate(self, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
            print(f"{Fore.CYAN}Client {args.node_id} starting evaluation...{Style.RESET_ALL}")
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
            f1 = f1_score(y_test, np.argmax(model.predict(X_test_scaled), axis=1), average='weighted')
            print(f"{Fore.CYAN}Client {args.node_id} evaluation done.{Style.RESET_ALL}")
            print(
                f"Client {args.node_id} Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                f"F1-score: {f1:.4f}"
            )
            # Log the evaluation metrics with the actual last epoch number and round number
            loss_writer.writerow(['test_loss', loss, self.current_epoch, self.round_number])
            accuracy_writer.writerow(['test_accuracy', accuracy, self.current_epoch, self.round_number])
            f1_writer.writerow(['test_f1_score', f1, self.current_epoch, self.round_number])
            return loss, len(X_test_scaled), {"accuracy": accuracy, "f1-score": f1}

    # Start Flower client and initiate communication with the Flower aggregation server
    print(f"Connecting to server at {args.address}:{args.port}")
    fl.client.start_numpy_client(
        server_address=f"{args.address}:{args.port}",
        client=Client(round_number=args.node_id)
    )
    print(f"{Fore.CYAN}Client {args.node_id} all processes finished.{Style.RESET_ALL}")

    # Close the CSV files
    loss_file.close()
    accuracy_file.close()
    f1_file.close()
