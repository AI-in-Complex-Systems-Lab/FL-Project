import pandas as pd
import random
import os

# Define the base directory as the current directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file within the datasets directory
file_path = os.path.join(base_dir, 'datasets', 'federated_datasets', 'client_train_data_3.csv')

# Load the CSV file
df = pd.read_csv(file_path)

# Feature manipulation - inject random noise into a specific percentage of rows
# 13 rows so far
features_to_manipulate = ['Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts' , 'Fwd Pkt Len Max' , 'Fwd Pkt Len Min' , 'Fwd Pkt Len Mean' , 'Fwd Pkt Len Std' , 'Bwd Pkt Len Max' , 'Bwd Pkt Len Min' , 'Bwd Pkt Len Mean' , 'Bwd Pkt Len Std' , 'Flow Byts/s']  # Replace with actual feature names
percent_to_manipulate = 0.7  # 70% of rows

# Get a sample of 70% of the indices
num_rows = len(df)
indices_to_change = random.sample(range(num_rows), int(num_rows * percent_to_manipulate))

# Apply random noise to the selected features
for idx in indices_to_change:
    for feature in features_to_manipulate:
        df.at[idx, feature] = df.at[idx, feature] + random.uniform(-20, 20)  # Adjust noise range as needed

# Save the modified dataset
output_file_path = os.path.join(base_dir, 'datasets', 'federated_datasets', 'client_train_data_3p.csv')
df.to_csv(output_file_path, index=False)

print("Feature manipulation complete. Saved to 'client_train_data_3p.csv'.")
