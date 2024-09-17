import flwr as fl
import utils
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from typing import Dict
import json
import socket

# Load the dataset
def load_data():
    df = pd.read_parquet("hf://datasets/19kmunz/iot-23-preprocessed/data/train-00000-of-00001-ad1ef30cd88c8d29.parquet")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    feature_columns = ['id.orig_p', 'id.resp_p', 'proto', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state', 'missed_bytes', 'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    target_column = 'label'

    X_train = train_df[feature_columns].values
    y_train = train_df[target_column].values
    X_test = test_df[feature_columns].values
    y_test = test_df[target_column].values

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), ['id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['proto', 'conn_state', 'history'])
        ]
    )

    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    X_train = torch.from_numpy(X_train.toarray().astype(np.float32))
    X_test = torch.from_numpy(X_test.toarray().astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
    y_test = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)

    return X_train, y_train, X_test, y_test, preprocessor

# Define the PyTorch model
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 8)
        self.linear2 = nn.Linear(8, 4)
        self.linear3 = nn.Linear(4, 1)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        logits = self.linear(x)
        logits = self.linear2(logits)
        logits = self.linear3(logits)
        y_predicted = torch.sigmoid(logits)
        return y_predicted

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    X_train, y_train, X_test, y_test, _ = load_data()  # Make sure load_data is available here

    def evaluate(server_round, parameters, config):
        utils.set_model_params(model, parameters)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            loss = nn.BCELoss()(y_pred, y_test).item()
            y_pred_cls = torch.round(y_pred)
            accuracy = (y_pred_cls == y_test).sum().item() / len(y_test)
        return loss, {"accuracy": accuracy}

    return evaluate

def get_ip_address():
    try:
        # Connect to an external server to determine the local IP address used for that connection
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # Google's public DNS server
            ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'  # Fallback to localhost
    return ip_address

ip_address = get_ip_address()
server_addr=ip_address + ':8080' 

# Write server address to a config file
config = {"server_address": server_addr}
with open("server_config.json", "w") as f:
    json.dump(config, f)

# Print the server address
print(f"Starting Flower server at {server_addr}")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, _ = load_data()
    n_features = X_train.shape[1]
    model = LogisticRegression(n_features)
    utils.set_initial_params(model)
    
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address=server_addr,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10),
    )
