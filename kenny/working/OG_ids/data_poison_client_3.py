import pandas as pd
import random

# Load the CSV file into a DataFrame
df = pd.read_csv('/Users/guest1/Documents/GitHub/FL-Project/kenny/working/OG_ids/datasets/federated_datasets/client_train_data_3.csv')  # Replace 'your_file.csv' with the actual file path

# Count how many times '3' appears in the 'y' column
count_three = df['y'].value_counts().get(3, 0)
print(f"Total number of 3s in 'y': {count_three}")

# Calculate 70% of the count
num_to_change = int(count_three * 0.9)
print(f"Number of 3s to change: {num_to_change}")

# Get the indices of all '3's
indices_of_three = df.index[df['y'] == 3].tolist()

# Randomly select 70% of the indices
indices_to_change = random.sample(indices_of_three, num_to_change)

# Change selected '3's to a random number between 0-10, excluding 3
for idx in indices_to_change:
    new_value = random.choice([i for i in range(0, 11) if i != 3])
    df.at[idx, 'y'] = new_value

# Count how many times '4' appears in the 'y' column
count_four = df['y'].value_counts().get(4, 0)
print(f"Total number of 4s in 'y': {count_four}")

# Calculate 70% of the count
num_to_change = int(count_four * 0.9)
print(f"Number of 4s to change: {num_to_change}")

# Get the indices of all '4's
indices_of_four = df.index[df['y'] == 4].tolist()

# Randomly select 70% of the indices
indices_to_change = random.sample(indices_of_four, num_to_change)

# Change selected '4's to a random number between 0-10, excluding 4
for idx in indices_to_change:
    new_value = random.choice([i for i in range(0, 11) if i != 4])
    df.at[idx, 'y'] = new_value

# Count how many times '5' appears in the 'y' column
count_five = df['y'].value_counts().get(5, 0)
print(f"Total number of 5s in 'y': {count_five}")

# Calculate 70% of the count
num_to_change = int(count_five * 0.9)
print(f"Number of 5s to change: {num_to_change}")

# Get the indices of all '5's
indices_of_five = df.index[df['y'] == 5].tolist()

# Randomly select 70% of the indices
indices_to_change = random.sample(indices_of_five, num_to_change)

# Change selected '5's to a random number between 0-10, excluding 5
for idx in indices_to_change:
    new_value = random.choice([i for i in range(0, 11) if i != 5])
    df.at[idx, 'y'] = new_value

# Count how many times '6' appears in the 'y' column
count_six = df['y'].value_counts().get(6, 0)
print(f"Total number of 6s in 'y': {count_six}")

# Calculate 70% of the count
num_to_change = int(count_six * 0.9)
print(f"Number of 6s to change: {num_to_change}")

# Get the indices of all '6's
indices_of_six = df.index[df['y'] == 6].tolist()

# Randomly select 70% of the indices
indices_to_change = random.sample(indices_of_six, num_to_change)

# Change selected '6's to a random number between 0-10, excluding 6
for idx in indices_to_change:
    new_value = random.choice([i for i in range(0, 11) if i != 6])
    df.at[idx, 'y'] = new_value

# Count how many times '7' appears in the 'y' column
count_seven = df['y'].value_counts().get(7, 0)
print(f"Total number of 6s in 'y': {count_seven}")

# Calculate 70% of the count
num_to_change = int(count_six * 0.9)
print(f"Number of 7s to change: {num_to_change}")

# Get the indices of all '7's
indices_of_seven = df.index[df['y'] == 7].tolist()

# Randomly select 70% of the indices
indices_to_change = random.sample(indices_of_seven, num_to_change)

# Change selected '7's to a random number between 0-10, excluding 7
for idx in indices_to_change:
    new_value = random.choice([i for i in range(0, 11) if i != 7])
    df.at[idx, 'y'] = new_value

# Count how many times '8' appears in the 'y' column
count_eight = df['y'].value_counts().get(8, 0)
print(f"Total number of 8s in 'y': {count_eight}")

# Calculate 70% of the count
num_to_change = int(count_eight * 0.9)
print(f"Number of 8s to change: {num_to_change}")

# Get the indices of all '8's
indices_of_eight = df.index[df['y'] == 8].tolist()

# Randomly select 70% of the indices
indices_to_change = random.sample(indices_of_eight, num_to_change)

# Change selected '8's to a random number between 0-10, excluding 8
for idx in indices_to_change:
    new_value = random.choice([i for i in range(0, 11) if i != 8])
    df.at[idx, 'y'] = new_value

#Save the modified DataFrame back to CSV
df.to_csv('client_train_data_3p.csv', index=False)  # Change the file name if needed

print("Modifications complete. Saved to 'client_train_data_3p.csv'.")
