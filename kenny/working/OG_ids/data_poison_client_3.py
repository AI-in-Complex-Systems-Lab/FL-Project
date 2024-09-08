import pandas as pd
import random

# Load the CSV file into a DataFrame
df = pd.read_csv('/Users/guest1/Documents/GitHub/FL-Project/kenny/working/OG_ids/datasets/federated_datasets/client_train_data_3.csv')  # Replace 'your_file.csv' with the actual file path

# Count how many times '6' appears in the 'y' column
count_six = df['y'].value_counts().get(6, 0)
print(f"Total number of 6s in 'y': {count_six}")

# Calculate 30% of the count
num_to_change = int(count_six * 0.7)
print(f"Number of 6s to change: {num_to_change}")

# Get the indices of all '6's
indices_of_six = df.index[df['y'] == 6].tolist()

# Randomly select 30% of the indices
indices_to_change = random.sample(indices_of_six, num_to_change)

# Change selected '6's to a random number between 0-10, excluding 6
for idx in indices_to_change:
    new_value = random.choice([i for i in range(0, 11) if i != 6])
    df.at[idx, 'y'] = new_value

# Optional: Save the modified DataFrame back to CSV
df.to_csv('modified_file.csv', index=False)  # Change the file name if needed

print("Modifications complete. Saved to 'PO_client_train_data_3.csv'.")
