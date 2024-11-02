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
	"""Send round number to client."""
	return {"server_round": server_round}

'''
def save_confusion_matrix(y_true, y_pred, labels):
    # Define the directory to save metrics
    metrics_dir = "metrics"
    
	# Create the directory if it doesn't exist
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create a confusion matrix display object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Plot the confusion matrix with specified colormap
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    
    # Adjust the ticks and labels if necessary
    ax.set_xticks(range(len(labels)))  # Set correct tick positions
    ax.set_yticks(range(len(labels)))  # Set correct tick positions
    ax.set_xticklabels(labels)         # Set correct tick labels
    ax.set_yticklabels(labels)         # Set correct tick labels
    
    # Save the confusion matrix figure
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(metrics_dir, "confusion_matrix.png"))
    plt.close()
'''

def save_metrics(round_number, train_loss, train_accuracy, eval_loss, eval_accuracy, f1):
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = os.path.join(metrics_dir, "5_client_global_metrics.csv")
    file_exists = os.path.isfile(metrics_file)

    with open(metrics_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["round", "train_loss", "train_accuracy", "eval_loss", "eval_accuracy", "f1_score"])
        writer.writerow([round_number, train_loss, train_accuracy, eval_loss, eval_accuracy, f1])

def get_evaluate_fn(model: Sequential):
    """Return an evaluation function for server-side evaluation."""

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        # Update model with the latest parameters
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
        f1 = f1_score(y_test, np.argmax(model.predict(X_test_scaled), axis=1), average='weighted')

        '''
        # Save the final confusion matrix at the end of the last round
        if server_round == args.rounds:
            y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
            save_confusion_matrix(y_test, y_pred, labels=[0, 1])  # Adjust labels if needed
        '''
        # Save metrics to file
        save_metrics(server_round, None, None, loss, accuracy, f1)

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
		sys.exit(f"Wrong IP address: {args.address}")
	if args.port < 0 or args.port > 65535:
		sys.exit(f"Wrong serving port: {args.port}")
	if args.rounds < 0:
		sys.exit(f"Wrong number of rounds: {args.rounds}")
	if not os.path.isdir(args.dataset):
		sys.exit(f"Wrong path to directory with datasets: {args.dataset}")


	# Load train and test data
	df_train = pd.read_csv(os.path.join(args.dataset, 'train_data.csv'))
	df_test = pd.read_csv(os.path.join(args.dataset, 'test_data.csv'))

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


	# Define a MLP model
	model = Sequential([
		InputLayer(input_shape=(X_test_scaled.shape[1],)),
		Dense(units=50, activation='relu'),
		Dropout(0.2),
		Dense(units=y_test_cat.shape[1], activation='softmax')
	])
      
	optimizer = Adam(learning_rate=0.001)
      
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	# Define a FL strategy
	strategy = fl.server.strategy.FedAvg(
		min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
		evaluate_fn=get_evaluate_fn(model),
		on_fit_config_fn=fit_round,
	)
      
	# Clear the metrics file at the start of the session
	metrics_dir = "metrics"
	os.makedirs(metrics_dir, exist_ok=True)
	metrics_file = os.path.join(metrics_dir, "5_client_global_metrics.csv")
	with open(metrics_file, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["round", "train_loss", "train_accuracy", "eval_loss", "eval_accuracy", "f1_score"])
    
	# Print the server address
	print(f"Starting Flower server at {args.address}:{args.port}")

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
