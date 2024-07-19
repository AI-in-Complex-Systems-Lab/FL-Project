from collections import OrderedDict
import flwr as fl 
import torch
from centralized import load_data, load_model, train, test
import json

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
net = load_model()
trainloader, testloader = load_data()

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    def fit(self, parameters, config):
        set_parameters(net, parameters)
        train(net, trainloader, epochs = 1)
        return self.get_parameters({}), len(trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader.dataset), {"accuracy": accuracy}
    
    
# Read server address from the config file
with open("server_config.json", "r") as f:
    config = json.load(f)    

server_addr = config["server_address"]

#ip_address = '169.226.53.20'  # here you should write the server ip-address
#server_addr=ip_address + ':8080'

# Print the server address
print(f"Starting Flower server at {server_addr}")

fl.client.start_client(
    server_address=server_addr,
    client=FlowerClient(),
)