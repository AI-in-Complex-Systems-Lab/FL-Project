import os
import random
import pandas as pd

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the dataset folder
dataset_dir = os.path.join(base_dir, 'datasets', 'federated_datasets')

# List all client files
client_files = [f for f in os.listdir(dataset_dir) if f.startswith('client_train_data_') and f.endswith('.csv')]

# Ensure there are at least 25 clients
assert len(client_files) >= 25, "There must be at least 25 client files to select from."

# Select 5 random clients
selected_clients = random.sample(client_files, 5)
print(f"Selected clients for poisoning: {selected_clients}")

# Poisoning logic
def poison_data(df):
    for x in range(0, 11, 2):  # Iterates through 0, 2, 4, 6, 8, 10
        # Count how many times 'x' appears in the 'y' column
        count_num = df['y'].value_counts().get(x, 0)
        print(f"Total number of {x}s in 'y': {count_num}")

        # Calculate 70% of the count
        num_to_change = int(count_num * 0.7)
        print(f"Number of {x}s to change: {num_to_change}")

        # Get the indices
        indices_of_num = df.index[df['y'] == x].tolist()

        # Randomly select 70% of the indices
        if num_to_change > 0:  # Ensure there are items to change
            indices_to_change = random.sample(indices_of_num, num_to_change)

            # Change selected 'x's to a random number between 0-10, excluding x
            for idx in indices_to_change:
                new_value = random.choice([i for i in range(0, 11) if i != x])
                df.at[idx, 'y'] = new_value

    return df

# Poison data for the selected clients
for client_file in selected_clients:
    # Load client data
    file_path = os.path.join(dataset_dir, client_file)
    df = pd.read_csv(file_path)
    
    # Poison the data
    poisoned_df = poison_data(df)
    
    # Save the poisoned data back
    poisoned_file_path = os.path.join(dataset_dir, f'p_{client_file}')
    poisoned_df.to_csv(poisoned_file_path, index=False)
    print(f"Poisoned data saved to: {poisoned_file_path}")

print("Poisoning complete.")

