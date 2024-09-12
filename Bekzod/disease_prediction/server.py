# server 
import flwr as fl
import numpy as np
import pandas as pd
import socket
import json
import os 
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

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


def evaluate(self, parameters, config):
    print("\n\n\n----------------  Test ----------------- ")
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    f1 = f1_score(y_test, np.argmax(model.predict(X_test_scaled), axis=1), average='weighted')
    
    # Save metrics to client file
    round_number = config.get("server_round", None)
    self.save_metrics(round_number, None, None, loss, accuracy, f1)
    
    print("Eval accuracy : ", accuracy)
    print(f"Evaluation results for client {self.client_id}: Loss = {loss}, Accuracy = {accuracy}, F1-Score = {f1}")
    return loss, len(X_test_scaled), {"accuracy": accuracy, "f-1 score": f1}
    
def save_metrics(self, round_number, train_loss, train_accuracy, eval_loss, eval_accuracy, f1):
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = self.metrics_file
    file_exists = os.path.isfile(metrics_file)

    with open(metrics_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["round", "train_loss", "train_accuracy", "eval_loss", "eval_accuracy", "f1_score"])
        # Handle None values
        writer.writerow([round_number, train_loss if train_loss is not None else 'N/A',
                         train_accuracy if train_accuracy is not None else 'N/A',
                         eval_loss if eval_loss is not None else 'N/A',
                         eval_accuracy if eval_accuracy is not None else 'N/A',
                         f1 if f1 is not None else 'N/A'])



def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

ip_address = get_ip_address()
server_addr=ip_address + ':8080' 

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
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3, 
),
)