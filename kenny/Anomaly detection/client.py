import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import glob
import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import anomaly
import flwr as fl

disable_progress_bar()


USE_FEDBN: bool = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Flower Client
class AnomalyClient(fl.client.NumPyClient):
    """Flower client implementing ... using PyTorch."""

    def __init__(
        self,
        model: anomaly.Net,
        trainloader: DataLoader,
        testloader: DataLoader,
        client_id,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.client_id = client_id
        self.approx_round_number = 1

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding
            # parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    '''def log_metric(self, metric_name, value, client_id):
        print(f"{metric_name},{value},{client_id},{self.approx_round_number}")'''

    def log_metric(self, metric_name, value, client_id):
        os.makedirs('metrics', exist_ok=True)
        filename = f"metrics/metrics_client_{client_id}.csv"
        with open(filename, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([metric_name, value, client_id, self.approx_round_number])

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        training_loss = anomaly.train(self.model, self.trainloader, epochs=100, device=DEVICE)
        self.log_metric("TRAIN_LOSS", training_loss, self.client_id)
        #self.approx_round_number += 1
        #print("TRAIN LOSS")
        #print(training_loss)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"training_loss": training_loss}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        testing_loss, accuracy = anomaly.test(self.model, self.testloader, device=DEVICE)
        self.log_metric("TEST_LOSS", testing_loss, self.client_id)
        self.approx_round_number += 1
        return testing_loss, len(self.testloader.dataset), {"testing_loss": testing_loss, "accuracy": accuracy}

def aggregate_metrics():
    all_files = glob.glob('metrics/*.csv')
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, names=['Metric', 'Value', 'ClientID', 'Round'])
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def plot_loss_per_round(metrics_df, metric_name, title, save_path):
    filtered_df = metrics_df[metrics_df['Metric'] == metric_name]
    pivot_df = filtered_df.pivot_table(index='Round', columns='ClientID', values='Value', aggfunc='mean')
    
    plt.figure(figsize=(10, 6))
    for column in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[column], marker='o', label=f'Client {column}')
        
    plt.title(title)
    plt.xlabel('Federated Round')
    plt.ylabel('Loss')
    plt.legend(title='Client ID')
    plt.grid(True)
    
    # Save the figure to a file
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    """Load data, start AnomalyClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--node-id", type=int, required=True, choices=range(0, 5))
    args = parser.parse_args()

    csv_path = './data/cell_data.csv'

    # Load data
    trainloader, testloader = anomaly.load_data(csv_path=csv_path, which_cell=args.node_id)

    # Load model
    model = anomaly.Net(input_dim=96).to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    #_ = model(next(iter(trainloader))["img"].to(DEVICE))

    # Start client
    #client = AnomalyClient(model, trainloader, testloader).to_client()
    client = AnomalyClient(model, trainloader, testloader, client_id=args.node_id).to_client()

    fl.client.start_client(server_address="[::]:8080", client=client)

    metrics_df = aggregate_metrics()

    # Generate and save the plots
    plot_loss_per_round(metrics_df, 'TRAIN_LOSS', 'Average Training Loss per Round', 'metrics/training_loss_per_round.png')
    plot_loss_per_round(metrics_df, 'TEST_LOSS', 'Average Testing Loss per Round', 'metrics/testing_loss_per_round.png')



if __name__ == "__main__":
    main()
