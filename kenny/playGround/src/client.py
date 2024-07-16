import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import flwr as fl
import utils
import json

# Define the model
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

# Load and preprocess data
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

class IoTClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):
        utils.set_model_params(self.model, parameters)
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(self.X_train)
        loss = self.criterion(y_pred, self.y_train)
        loss.backward()
        self.optimizer.step()
        return utils.get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        utils.set_model_params(self.model, parameters)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test)
            loss = self.criterion(y_pred, self.y_test).item()
            y_pred_cls = torch.round(y_pred)
            accuracy = (y_pred_cls == self.y_test).sum().item() / len(self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, preprocessor = load_data()
    n_features = X_train.shape[1]
    model = LogisticRegression(n_features)
    utils.set_initial_params(model)

    client = IoTClient(model, X_train, y_train, X_test, y_test)
    # Read server address from the config file
    with open("server_config.json", "r") as f:
     config = json.load(f)    

    server_addr = config["server_address"]

    # Print the server address
    print(f"Starting Flower server at {server_addr}")
    fl.client.start_client(server_address=server_addr, client=client)

