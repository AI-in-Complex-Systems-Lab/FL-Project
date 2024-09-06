from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


import flwr as fl


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
   trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
   trainset = CIFAR10("./data", train = True, download = True, transform = trf)
   testset = CIFAR10("./data", train = False, download = True, transform = trf)
   return DataLoader(trainset, batch_size = 32, shuffle = True), DataLoader(testset)


def train(net, trainloader, epochs):
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum = 0.9)
   for _ in range(epochs):
       for images, labels in trainloader:
           optimizer.zero_grad()
           criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
           optimizer.step()


def test(net, testloader):
   criterion = torch.nn.CrossEntropyLoss()
   correct, total, loss = 0, 0, 0.0
   with torch.no_grad():
       for images, labels in testloader:
           outputs = net(images.to(DEVICE))
           loss += criterion(outputs, labels.to(DEVICE)).item()
           total += labels.size(0)
           correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
   return loss / len(testloader.dataset), correct / total


class Net(nn.Module):
   def __init__(self) -> None:
       super(Net, self).__init__()
       self.conv1 = nn.Conv2d(3, 6, 5)
       self.pool = nn.MaxPool2d(2, 2)
       self.conv2 = nn.Conv2d(6, 16, 5)
       self.fc1 = nn.Linear(16 * 5 * 5, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, 10)


   def forward(self, x):
       x = self.pool(F.relu(self.conv1(x)))
       x = self.pool(F.relu(self.conv2(x)))
       x = x.view(-1, 16 * 5 * 5)
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       return self.fc3(x)


# Load model and data
def load_model():
   return Net().to(DEVICE)


if __name__ == "__main__":
   net = load_model()
   trainloader, testloader = load_data()


def set_parameters(self, parameters):
       params_dict = zip(net.state_dict().keys(), parameters)
       state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
       net.load_state_dict(state_dict, strict=True)


class Client(fl.client.NumPyClient):
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


# Start the client
fl.client.start_client(
   server_address="127.0.0.1:8080",
   client=Client(),
   )
