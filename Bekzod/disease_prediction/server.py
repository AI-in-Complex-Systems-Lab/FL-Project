# server 
import flwr as fl
import numpy as np
import pandas as pd
import socket
import json
import os 
import sys

def get_ip_address():
    try:
        # Connect to an external server to determine the local IP address used for that connection
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # Google's public DNS server
            ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'  # Fallback to localhost
    return ip_address

# Split train_data.csv into three equal parts


'''
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
strategy = SaveModelStrategy()
'''

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

ip_address = get_ip_address()
server_addr=ip_address + ':8080' 


def fit_config(rnd: int):
    return {
        "server_round": rnd  # Pass the round number
    }

# Write server address to a config file
config = {"ip_address": ip_address,"server_address": server_addr}
with open("server_config.json", "w") as f:
	json.dump(config, f)

# Print the server address
print(f"Starting Flower server at {server_addr}")

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = server_addr, 
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn = weighted_average, 
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5, 
        on_fit_config_fn=fit_config,
),
)