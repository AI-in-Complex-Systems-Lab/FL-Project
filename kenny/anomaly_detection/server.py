"""Flower server example."""

#import matplotlib.pyplot as plt

from typing import List, Tuple, Optional, Dict, Union
import flwr as fl
import numpy as np
import socket
import json

from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, Scalar, Metrics

NUM_OF_CLIENTS = 6
NUM_ROUNDS = 10

def get_ip_address():
    try:
        # Connect to an external server to determine the local IP address used for that connection
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # Google's public DNS server
            ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'  # Fallback to localhost
    return ip_address

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    fraction_evaluate=1.0, # 100% of clients participate in the evaluation
    fraction_fit=1.0, # 100% of clients participate in the training
    min_fit_clients=NUM_OF_CLIENTS, # Minimum of 6 clients for training
    min_evaluate_clients=NUM_OF_CLIENTS, # Minimum of 6 clients for evaluation
    min_available_clients=NUM_OF_CLIENTS, # Minimum of 6 clients to start training
)

ip_address = get_ip_address()
server_addr=ip_address + ':8080' 

# Write server address to a config file
config = {"server_address": server_addr}
with open("server_config.json", "w") as f:
    json.dump(config, f)

# Print the server address
print(f"Starting Flower server at {server_addr}")

#ip_address = '169.226.53.20'
#server_addr=ip_address + ':8080'

# Start Flower server
fl.server.start_server(
    server_address=server_addr,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)