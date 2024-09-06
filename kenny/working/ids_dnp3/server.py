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
import flwr as fl
from colorama import Fore, Style
import socket
import json

def get_ip_address():
    try:
        # Connect to an external server to determine the local IP address used for that connection
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # Google's public DNS server
            ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'  # Fallback to localhost
    return ip_address

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def get_evaluate_fn(model: Sequential):
    """Return an evaluation function for server-side evaluation."""

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        print(f"{Fore.CYAN}Server starting evaluation...{Style.RESET_ALL}")
        # Update model with the latest parameters
        model.set_weights(parameters)
        print(f"{Fore.CYAN}Server model weights set.{Style.RESET_ALL}")

        loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
        print(f"{Fore.CYAN}Server evaluation done.{Style.RESET_ALL}")

        f1 = f1_score(y_test, np.argmax(model.predict(X_test_scaled), axis=1), average='weighted')
        print(f"{Fore.CYAN}Server F1-score computed.{Style.RESET_ALL}")

        return loss, {"accuracy": accuracy, "f1-score": f1}

    return evaluate

# Define the base directory as the current directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to ids_dnp3 directory
ids_dnp3_federated_datasets_path = os.path.join(base_dir, 'datasets', 'federated_datasets')

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Flower aggregator server implementation')
    parser.add_argument("-a", "--address", help="IP address", default=get_ip_address())
    parser.add_argument("-p", "--port", help="Serving port", default=8080, type=int)
    parser.add_argument("-r", "--rounds", help="Number of training and aggregation rounds", default=20, type=int)
    parser.add_argument("-d", "--dataset", help="dataset directory", default=ids_dnp3_federated_datasets_path)
    args = parser.parse_args()

    try:
        ipaddress.ip_address(args.address)
    except ValueError:
        sys.exit(f"{Fore.RED}Wrong IP address: {args.address}{Style.RESET_ALL}")
    if args.port < 0 or args.port > 65535:
        sys.exit(f"{Fore.RED}Wrong serving port: {args.port}{Style.RESET_ALL}")
    if args.rounds < 0:
        sys.exit(f"{Fore.RED}Wrong number of rounds: {args.rounds}{Style.RESET_ALL}")
    if not os.path.isdir(args.dataset):
        sys.exit(f"{Fore.RED}Wrong path to directory with datasets: {args.dataset}{Style.RESET_ALL}")

    # Load train and test data
    print(f"{Fore.CYAN}Server loading train and test data...{Style.RESET_ALL}")
    df_train = pd.read_csv(os.path.join(args.dataset, 'train_data.csv'))
    df_test = pd.read_csv(os.path.join(args.dataset, 'test_data.csv'))
    print(f"{Fore.CYAN}Server data loaded.{Style.RESET_ALL}")

    # Convert data to arrays
    X_train = df_train.drop(['y'], axis=1).to_numpy()
    X_test = df_test.drop(['y'], axis=1).to_numpy()

    # Convert test data labels to one-hot-vectors
    y_test = df_test['y'].to_numpy()
    y_test_cat = to_categorical(y_test)

    # Scale test data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"{Fore.CYAN}Server data scaled.{Style.RESET_ALL}")

    # Define a MLP model
    model = Sequential([
        InputLayer(input_shape=(X_test_scaled.shape[1],)),
        Dense(units=50, activation='relu'),
        Dropout(0.2),
        Dense(units=y_test_cat.shape[1], activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(f"{Fore.CYAN}Server model compiled.{Style.RESET_ALL}")

    # Define a FL strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    
    # Print the server address
    print(f"{Fore.CYAN}Starting Flower server at {args.address}:{args.port}{Style.RESET_ALL}")

    server_addr = f"{args.address}:{args.port}"

    # Write server address to a config file
    config = {"ip_address": args.address,"server_address": server_addr}
    with open("server_config.json", "w") as f:
     json.dump(config, f)

    # Start Flower aggregation and distribution server
    fl.server.start_server(
        server_address=f"{args.address}:{args.port}",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )
    print(f"{Fore.CYAN}All process finished.{Style.RESET_ALL}")