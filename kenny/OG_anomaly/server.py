"""Flower server example."""

#import matplotlib.pyplot as plt

from typing import List, Tuple, Optional, Dict, Union
import flwr as fl
import numpy as np

from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, Scalar, Metrics

NUM_OF_CLIENTS = 5
NUM_ROUNDS = 10


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
    min_fit_clients=NUM_OF_CLIENTS, # Minimum of 5 clients for training
    min_evaluate_clients=NUM_OF_CLIENTS, # Minimum of 5 clients for evaluation
    min_available_clients=NUM_OF_CLIENTS, # Minimum of 5 clients to start training
)

# Start Flower server
fl.server.start_server(
    server_address="[::]:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)