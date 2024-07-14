# This is a supervised learning, netural network apporach.

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler  # for feature scaling
from sklearn.model_selection import train_test_split  # for train/test split
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the dataset from the Parquet file
df = pd.read_parquet("hf://datasets/19kmunz/iot-23-preprocessed/data/train-00000-of-00001-ad1ef30cd88c8d29.parquet")

# Splitting the dataset into train and test sets (80/20 split)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create a directory named 'data' if it doesn't exist
data_folder = 'data'
os.makedirs(data_folder, exist_ok=True)

# Define file paths for train and test CSV files within the 'data' folder
train_csv_path = os.path.join(data_folder, 'train.csv')
test_csv_path = os.path.join(data_folder, 'test.csv')

# Save the train and test sets to CSV files in the 'data' folder
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

# Optionally, you can print the sizes of the individual datasets
# Calculate sizes
total_size = len(train_df) + len(test_df)
# Print the total size and percentages
print(f"Total data size: {total_size}")
print(f"Training data size: {len(train_df)}")
print(f"Testing data size: {len(test_df)}")

# Prepare data

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
print(f'number of samples: {n_samples}, number of features: {n_features}')

# split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale data

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convert to tensors

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape y tensors

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# Create model
# f = wx + b, sigmoid at the end

class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = LogisticRegression(n_features)

# Loss and optimizer

learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop

num_epochs = 100

for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    
    # backward pass
    loss.backward()
    
    # updates
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# evaluate model

with torch.no_grad():
    y_predicted = model(X_test)  # no need to call model.forward()
    y_predicted_cls = y_predicted.round()   # round off to nearest class
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])  # accuracy
    print(f'accuracy = {acc:.4f}')