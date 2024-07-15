# This is a supervised learning, netural network apporach.

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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

# Calculate sizes
total_size = len(train_df) + len(test_df)
# Print the total size and percentages
print(f"Total data size: {total_size}")
# 80% train, 20% test
print(f"Training data size: {len(train_df)}")
print(f"Testing data size: {len(test_df)}")

# Prepare data for training
# Define feature columns and target column
feature_columns = ['id.orig_p', 'id.resp_p', 'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state', 'missed_bytes', 'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
target_column = 'label'

# Extract features and target
X_train = train_df[feature_columns].values
y_train = train_df[target_column].values
X_test = test_df[feature_columns].values
y_test = test_df[target_column].values

# Encode target labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Define the preprocessor to handle numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), ['id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['proto', 'service', 'conn_state', 'history'])
    ]
)

# Fit and transform the training data
X_train = preprocessor.fit_transform(train_df)
# Transform the test data
X_test = preprocessor.transform(test_df)

# Convert sparse matrix to dense numpy array
X_train = X_train.toarray()
X_test = X_test.toarray()

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)

class LogisticRegression(nn.Module):

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        nn.init.xavier_uniform_(self.linear.weight)  # Initialize weights

    def forward(self, x):
        logits = self.linear(x)
        y_predicted = torch.sigmoid(logits)
        return y_predicted
    
# Initialize model
n_features = X_train.shape[1]
model = LogisticRegression(n_features)

# Loss and optimizer

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training loop

num_epochs = 1000

for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    
   # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        # Evaluate model
        with torch.no_grad():
            y_predicted = model(X_test)
            y_predicted_cls = torch.round(y_predicted)  # round off to nearest class
            accuracy = (y_predicted_cls == y_test).sum().item() / len(y_test)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')