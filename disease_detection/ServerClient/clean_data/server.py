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
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import flwr as fl
import socket
import json
import csv


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
    return {"server round": server_round}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

# Create strategy and run server
strategy = fl.server.strategy.FedAvg(
	min_fit_clients=3,
    min_evaluate_clients=3,
     min_available_clients=3,
	evaluate_fn=get_evaluate_fn(model),
	on_fit_config_fn=fit_round,
	)

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

ip_address = get_ip_address()
server_addr=ip_address + ':8080' 

# Write server address to a config file


# Print the server address
print(f"Starting Flower server at {args.}")
server_addr = f"{args.address}:{args.port}"

config = {"ip_address": args.address, "server_address": server_addr}
with open("server_config.json", "w") as f:
    json.dump(config, f)


# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = f"{args.address}:{args.port}", 
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
)
