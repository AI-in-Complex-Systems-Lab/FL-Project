import argparse
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import flwr as fl
import utils

# Define the columns
feature_columns = ['id.orig_p', 'id.resp_p', 'proto', 'service', 'duration', 'orig_bytes',
                   'resp_bytes', 'conn_state', 'missed_bytes', 'history', 'orig_pkts',
                   'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
target_column = 'label'
categorical_columns = ['proto', 'service', 'conn_state', 'history']
numerical_columns = [col for col in feature_columns if col not in categorical_columns]

# Load the dataset
def load_dataset(partition_id=None):
    # Replace with actual path to your Parquet file
    df = pd.read_parquet("hf://datasets/19kmunz/iot-23-preprocessed/data/train-00000-of-00001-ad1ef30cd88c8d29.parquet")
    if partition_id is not None:
        df = df[df['partition_id'] == partition_id]  # Adjust partitioning logic as needed
    X = df[feature_columns]
    y = df[target_column]
    return X, y

if __name__ == "__main__":
    N_CLIENTS = 10

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=False,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    partition_id = args.partition_id

    # Load the partition data
    X, y = load_dataset(partition_id)

    # Preprocess the data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ]
    )
    X = preprocessor.fit_transform(X)

    # Split the data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class IoTClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_client(
        server_address="0.0.0.0:8080", client=IoTClient().to_client()
    )
