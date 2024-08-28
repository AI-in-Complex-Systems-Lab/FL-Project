import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Net(nn.Module):
    """Dense Neural Network for Anomaly Detection."""
    def __init__(self, input_dim) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def load_data(csv_path: str, which_cell: int = 0, window_len: int = 96) -> tuple:
    dataset_train = pd.read_csv(csv_path)
    train_data = dataset_train.values

    # Initialize arrays to store anomaly and load data
    anomaly_data = np.zeros((5, 3, 5, 16608))
    load_data = np.zeros((5, 3, 5, 16608))

    # Splitting the dataset into cell-, category-, and device-specific data
    for cell in range(5):
        for category in range(3):
            for device in range(5):
                rows = reduce(np.intersect1d, (np.where(train_data[:, 1] == cell),
                                               np.where(train_data[:, 2] == category),
                                               np.where(train_data[:, 3] == device)))
                load_data[cell, category, device] = train_data[rows, 4]
                anomaly_data[cell, category, device] = train_data[rows, 5]

    # Aggregate load and anomaly data for the selected cell
    load_set = np.sum(np.sum(load_data[which_cell, :, :, :], axis=0), axis=0)
    anomaly_set = np.sum(np.sum(anomaly_data[which_cell, :, :, :], axis=0), axis=0)
    anomaly_set[anomaly_set != 0] = 1

    sc = MinMaxScaler(feature_range=(0, 1))
    load_set_scaled = sc.fit_transform(load_set.reshape(-1, 1))

    X, y = [], []
    for i in range(window_len, len(load_set_scaled)):
        X.append(load_set_scaled[i-window_len:i, 0])
        y.append(anomaly_set[i])
    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32)

    return train_loader, test_loader

def train(net: Net, trainloader: DataLoader, epochs: int, device: torch.device) -> float:
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.to(device)
    net.train()
    
    total_loss = 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    average_loss = total_loss / (len(trainloader.dataset) * epochs)
    return average_loss

'''def train(net: Net, trainloader: DataLoader, epochs: int, device: torch.device) -> None:
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    net.to(device)
    net.train()

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')'''

'''def test(net: Net, testloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.BCELoss()
    net.eval()
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    average_loss = test_loss / len(testloader)
    return average_loss, accuracy'''

def test(net: Net, testloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.BCELoss()
    net.eval()
    
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    average_loss = total_loss / len(testloader.dataset)
    accuracy = correct / total
    return average_loss, accuracy

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = './data/cell_data.csv'

    print("Centralized PyTorch training")
    print("Load data")

    trainloader, testloader = load_data(csv_path)
    net = Net(input_dim=96).to(DEVICE)
    net.eval()

    print("Start training")
    train(net=net, trainloader=trainloader, epochs=100, device=DEVICE)

    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()
