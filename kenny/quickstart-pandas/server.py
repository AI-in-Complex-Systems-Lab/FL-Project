from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy


# This class defines a custom federated learning strategy by inheriting from the Strategy base class.
class FedAnalytics(Strategy):

    # This method initializes the parameters before training starts.
    # It returns None, indicating no initial parameters are set.
    def initialize_parameters(
        self, client_manager: Optional[ClientManager] = None
    ) -> Optional[Parameters]:
        return None

    # Configures the training (fit) instructions to be sent to the clients.
    # Samples two clients using the client_manager.
    # Returns a list of tuples, each containing a client and the corresponding fit instructions.
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(num_clients=2, min_num_clients=2)
        return [(client, fit_ins) for client in clients]

    # Aggregates the fit results from the clients.
    # Converts parameters from the clients into numpy arrays.
    # Aggregates the "length" and "width" histograms.
    # Concatenates the aggregated results into a single numpy array.
    # Converts the numpy array back to Flower parameters format and returns them.
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Get results from fit
        # Convert results
        values_aggregated = [
            (parameters_to_ndarrays(fit_res.parameters)) for _, fit_res in results
        ]
        length_agg_hist = 0
        width_agg_hist = 0
        for val in values_aggregated:
            length_agg_hist += val[0]
            width_agg_hist += val[1]

        ndarr = np.concatenate(
            (["Length:"], length_agg_hist, ["Width:"], width_agg_hist)
        )
        return ndarrays_to_parameters(ndarr), {}

    # Evaluates the aggregated parameters.
    # Converts the parameters to numpy arrays and extracts their values.
    # Returns a dummy evaluation metric (0) and the aggregated histograms.
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        agg_hist = [arr.item() for arr in parameters_to_ndarrays(parameters)]
        return 0, {"Aggregated histograms": agg_hist}

    # Placeholder for configuring the evaluation instructions to be sent to the clients.
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        pass
    
    # Placeholder for aggregating the evaluation results from the clients.
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        pass


# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=FedAnalytics(),
)
