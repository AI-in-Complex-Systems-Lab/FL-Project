import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import flwr as fl

from flwr_datasets import FederatedDataset


column_names = ["sepal_length", "sepal_width"]

# This function computes the histogram of a specified column in a pandas DataFrame and returns the frequencies.
def compute_hist(df: pd.DataFrame, col_name: str) -> np.ndarray:
    freqs, _ = np.histogram(df[col_name])
    return freqs


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X: pd.DataFrame):
        self.X = X

    # It computes histograms for each column in self.X and returns a tuple containing the list of histograms, the number of samples, and an empty dictionary.
    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        hist_list = []
        # Execute query locally
        for c in self.X.columns:
            hist = compute_hist(self.X, c)
            hist_list.append(hist)
        return (
            hist_list,
            len(self.X),
            {},
        )


if __name__ == "__main__":
    N_CLIENTS = 2

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the partition id of artificially partitioned datasets.",
    )
    args = parser.parse_args()
    partition_id = args.partition_id

    # Load the partition data
    fds = FederatedDataset(dataset="hitorilabs/iris", partitioners={"train": N_CLIENTS})

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    # Use just the specified columns
    X = dataset[column_names]

    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(X).to_client(),
    )
