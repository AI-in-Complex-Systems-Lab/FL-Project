import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
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
numerical_features = ['id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
categorical_features = ['proto', 'service', 'conn_state', 'history']
target_column = 'label'

train_df = train_df.dropna()
test_df = test_df.dropna()

# Extract features and target
X_train_num = train_df[numerical_features].values
X_train_cat = train_df[categorical_features].values
y_train = train_df[target_column].values
X_test_num = test_df[numerical_features].values
X_test_cat = test_df[categorical_features].values
y_test = test_df[target_column].values

# Encode target labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Handle missing values in numerical features
imputer = SimpleImputer(strategy='mean')
X_train_num = imputer.fit_transform(X_train_num)
X_test_num = imputer.transform(X_test_num)

# Scale numerical features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# One-hot encode categorical features
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
X_train_cat = one_hot_encoder.fit_transform(X_train_cat).toarray()
X_test_cat = one_hot_encoder.transform(X_test_cat).toarray()

# Combine numerical and categorical features
X_train = np.hstack((X_train_num, X_train_cat))
X_test = np.hstack((X_test_num, X_test_cat))

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)

class LogisticRegression(nn.Module):

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        #self.linear2 = nn.Linear(32, 16)
        #self.linear3 = nn.Linear(16, 1)
        nn.init.xavier_uniform_(self.linear.weight)  # Initialize weights

    def forward(self, x):
        logits = self.linear(x)
        #logits = self.linear2(logits)
        #logits = self.linear3(logits)
        y_predicted = torch.sigmoid(logits)
        return y_predicted
    
# Initialize model
n_features = X_train.shape[1]
model = LogisticRegression(n_features)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50

for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    
    # Backward pass
    loss.backward()
    
    # Updates
    optimizer.step()
    
    # Zero gradients
    optimizer.zero_grad()
    
    if (epoch+1) % 5 == 0:
        y_predicted = model(X_test)
        # Evaluate model
        with torch.no_grad():
            y_predicted_cls = torch.round(y_predicted)  # Round off to nearest class
            accuracy = (y_predicted_cls == y_test).sum().item() / len(y_test)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
