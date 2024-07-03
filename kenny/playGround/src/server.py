import flwr as fl
import socket
import json

def get_ip_address():
    try:
        # Connect to an external server to determine the local IP address used for that connection
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # Google's public DNS server
            ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'  # Fallback to localhost
    return ip_address

# Create an instance of the model and get the parameters
params = get_parameters(Net())

# Pass parameters to the Strategy for server-side parameter initialization
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.3,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(params),
)


ip_address = get_ip_address()
server_addr=ip_address + ':8080' 

# Write server address to a config file
config = {"server_address": server_addr}
with open("server_config.json", "w") as f:
    json.dump(config, f)

# Print the server address
print(f"Starting Flower server at {server_addr}")

fl.server.start_server(
    server_address=server_addr,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn = weighted_average,
    ),
)