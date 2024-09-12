# import argparse
# import ipaddress
# import os
# import sys
# from typing import Dict
# import numpy as np
# import pandas as pd
# from sklearn.metrics import f1_score
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# import flwr as fl
# import socket
# import json
# import csv



# import argparse

# parser = argparse.ArgumentParser(description='Flower server')
# parser.add_argument('--address', type=str, default='169.226.237.129', help='Server IP address')
# parser.add_argument('--port', type=int, default=8080, help='Server port')
# parser.add_argument('--rounds', type=int, default=20, help='Number of rounds')
# args = parser.parse_args()


# import socket

# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.bind(('0.0.0.0', 8080))
# server_socket.listen(1)
# print("Server listening on port 8080")

# while True:
#     client_socket, addr = server_socket.accept()
#     print(f"Connection from {addr}")
#     client_socket.sendall(b"Hello from server")
#     client_socket.close()


# ip_address = args.address
# server_addr = f"{ip_address}:{args.port}"

# config = {"ip_address": args.address, "server_address": server_addr}
# with open("server_config.json", "w") as f:
#     json.dump(config, f)




# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# import pickle
# import os

# # Load and preprocess the dataset on a server
# data = pd.read_csv("Encoded_HeartDisease.csv")

# # Encode categorical features
# categorical_columns = ["HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", 
#                        "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", 
#                        "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
# label_encoder = LabelEncoder()
# for col in categorical_columns:
#     data[col] = label_encoder.fit_transform(data[col])

# # Split data into train and test sets
# X = data.drop("HeartDisease", axis=1)
# y = data["HeartDisease"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Save batches of training data to be sent to JetBots
# batch_size = 64
# for i in range(0, len(X_train), batch_size):
#     batch = (X_train[i:i + batch_size], y_train[i:i + batch_size])
#     with open(f'batch_{i//batch_size}.pkl', 'wb') as f:
#         pickle.dump(batch, f)

# print("Training batches saved for JetBots to load individually.")



# def get_ip_address():
#     try:
#         # Connect to an external server to determine the local IP address used for that connection
#         with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
#             s.connect(("8.8.8.8", 80))  # Google's public DNS server
#             ip_address = s.getsockname()[0]
#     except Exception:
#         ip_address = '127.0.0.1'  # Fallback to localhost
#     return ip_address


# def fit_round(server_round: int) -> Dict:
#     return {"server round": server_round}

# class SaveModelStrategy(fl.server.strategy.FedAvg):
#     def aggregate_fit(
#         self,
#         rnd,
#         results,
#         failures
#     ):
#         aggregated_weights = super().aggregate_fit(rnd, results, failures)
#         if aggregated_weights is not None:
#             # Save aggregated_weights
#             print(f"Saving round {rnd} aggregated_weights...")
#             np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
#         return aggregated_weights

# # Create strategy and run server
# strategy = fl.server.strategy.FedAvg(
# 	min_fit_clients=3,
#     min_evaluate_clients=3,
#      min_available_clients=3,
# 	evaluate_fn=get_evaluate_fn(model),
# 	on_fit_config_fn=fit_round,
# 	)

# def weighted_average(metrics):
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     return {"accuracy": sum(accuracies) / sum(examples)}

# ip_address = get_ip_address()
# server_addr=ip_address + ':8080' 

# # Write server address to a config file


# # Print the server address

# server_addr = f"{args.address}:{args.port}"
# print(f"Starting Flower server at {args.server_addr}")

# config = {"ip_address": args.address, "server_address": server_addr}
# with open("server_config.json", "w") as f:
#     json.dump(config, f)


# # Start Flower server for three rounds of federated learning
# fl.server.start_server(
#         server_address = f"{args.address}:{args.port}", 
#         config=fl.server.ServerConfig(num_rounds=args.rounds),
#         strategy=strategy,
# )



import argparse
import ipaddress
import json
import os
import pickle
import socket
import numpy as np
import pandas as pd
import flwr as fl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def get_ip_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("0.0.0.0", 80))  # Google's public DNS server
            return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'  # Fallback to localhost

def fit_round(server_round: int) -> dict:
    return {"server round": server_round}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

def load_and_preprocess_data():
    data = pd.read_csv("Encoded_HeartDisease.csv")
    categorical_columns = [
        "HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
        "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity",
        "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"
    ]
    

import os

def save_batches(X_train, y_train, batch_size=64):
    # Create the 'batches' directory if it doesn't exist
    os.makedirs('batches', exist_ok=True)
    
    for i in range(0, len(X_train), batch_size):
        batch = (X_train[i:i + batch_size], y_train[i:i + batch_size])
        # Save each batch to the 'batches' directory
        batch_filename = os.path.join('batches', f'batch_{i//batch_size}.pkl')
        with open(batch_filename, 'wb') as f:
            pickle.dump(batch, f)
    
    print("Training batches saved in 'batches' directory.")


def main():
    parser = argparse.ArgumentParser(description='Flower server')
    parser.add_argument('--address', type=str, default='169.226.237.129', help='Server IP address')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--rounds', type=int, default=20, help='Number of rounds')
    
    args = parser.parse_args()

    # Save configuration
    config = {"ip_address": args.address, "server_address": f"{args.address}:{args.port}"}
    with open("server_config.json", "w") as f:
        json.dump(config, f)

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    save_batches(X_train, y_train)

    # Define strategy
    strategy = SaveModelStrategy(
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=fit_round
    )

    # Start Flower server
    print(f"Starting Flower server at {args.address}:{args.port}")
    fl.server.start_server(
        server_address=f"{args.address}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
