import pandas as pd
import random
import os

# Define the base directory as the current directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file within the datasets directory
file_path = os.path.join(base_dir, 'datasets', 'federated_datasets', 'client_train_data_3.csv')

# Load the CSV file
df = pd.read_csv(file_path)

# Define the features for outlier injection and the percentage of rows to alter
# 13 rows so far
outlier_features = ['Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts' , 'Fwd Pkt Len Max' , 'Fwd Pkt Len Min' , 'Fwd Pkt Len Mean' , 'Fwd Pkt Len Std' , 'Bwd Pkt Len Max' , 'Bwd Pkt Len Min' , 'Bwd Pkt Len Mean' , 'Bwd Pkt Len Std' , 'Flow Byts/s']  # Replace with actual feature names in the dataset
outlier_percent = 0.1  # 60% of rows
extreme_value = 100000  # Extreme value to introduce as outliers

# Calculate the number of rows to change based on the outlier percentage
num_rows = len(df)
num_outliers = int(num_rows * outlier_percent)

# Get random indices for injecting outliers
outlier_indices = random.sample(range(num_rows), num_outliers)

# Set selected features to the extreme value for the outlier rows
for idx in outlier_indices:
    for feature in outlier_features:
        df.at[idx, feature] = extreme_value

# Define the path for saving the modified file in the same directory
output_file_path = os.path.join(base_dir, 'datasets', 'federated_datasets', 'client_train_data_3p.csv')

# Save the modified DataFrame back to CSV
df.to_csv(output_file_path, index=False)

print("Outlier injection complete. Saved to 'client_train_data_3p.csv'.")
