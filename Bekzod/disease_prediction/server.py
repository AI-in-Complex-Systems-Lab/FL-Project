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
def save_metrics(round_number, train_loss, train_accuracy, eval_loss, eval_accuracy, f1):
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = os.path.join(metrics_dir, "global_metrics.csv")
    file_exists = os.path.isfile(metrics_file)

    with open(metrics_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["round", "train_loss", "train_accuracy", "eval_loss", "eval_accuracy", "f1_score"])
        # Handle None values
        writer.writerow([round_number, train_loss, train_accuracy, eval_loss, eval_accuracy, f1])

def get_evaluate_fn(model: Sequential):
    """Return an evaluation function for server-side evaluation."""

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        print("\n\n\n----------------  Test ----------------- ")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
        f1 = f1_score(y_test, np.argmax(model.predict(X_test_scaled), axis=1), average='weighted')
    
        # Save metrics to client file
        save_metrics(server_round, None, None, loss, accuracy, f1)
    
        return loss, {"accuracy": accuracy, "f1-score": f1}
    return evaluate
    
base_dir = os.path.dirname(os.path.abspath(__file__))
disease_prediction_federated_datasets_path = os.path.join(base_dir, 'datasets', 'federated_datasets')



if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description='Flower aggregator server implementation')
	parser.add_argument("-a", "--address", help="IP address", default=get_ip_address())
	parser.add_argument("-p", "--port", help="Serving port", default=8080, type=int)
	parser.add_argument("-r", "--rounds", help="Number of training and aggregation rounds", default=20, type=int)
	parser.add_argument("-d", "--dataset", help="dataset directory", default="/Users/guest2/Desktop/FL-Project/Bekzod/disease_prediction/dataset")
	args = parser.parse_args()
        
try:
	ipaddress.ip_address(args.address)
except ValueError:
	sys.exit(f"Wrong IP address: {args.address}")
if args.port < 0 or args.port > 65535:
	sys.exit(f"Wrong serving port: {args.port}")
if args.rounds < 0:
	sys.exit(f"Wrong number of rounds: {args.rounds}")
if not os.path.isdir(args.dataset):
	sys.exit(f"Wrong path to directory with datasets: {args.dataset}")
      
	df_train = pd.read_csv(os.path.join(args.dataset, 'train_data.csv'))
	df_test = pd.read_csv(os.path.join(args.dataset, 'test_data.csv'))
      
X_train = df_train.drop(['HeartDisease']).to_numpy()
y_train = df_train['HeartDisease'].to_numpy()
X_test = df_test.drop(columns=['HeartDisease']).to_numpy()
y_test = df_test['HeartDisease'].to_numpy()

    
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