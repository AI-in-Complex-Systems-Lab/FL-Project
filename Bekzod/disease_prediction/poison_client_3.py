import pandas as pd
import random 
import numpy as np


df = pd.read_csv('/Users/guest2/Desktop/FL-Project/Bekzod/disease_prediction/dataset/client_train_data_3.csv')

# df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))



# print("All zeros at the end been changed to one and vice versa")

# num_nines_found = 0

# # Define a function to replace 9 with a random number between 1 and 10, excluding 9
# def replace_nine(x):
#     global num_nines_found
#     try:
#         # Convert to string to check for '9'
#         str_x = str(x)
#         if '9' in str_x:
#             num_nines_found += str_x.count('9')
#             # Replace '9' with a random number between 1 and 10, excluding 9
#             new_value = np.random.choice([i for i in range(1, 11) if i != 9])
#             return new_value
#         else:
#             return x
#     except:
#         return x

# Apply the function to the entire DataFrame
# df = df.map(replace_nine)

# print(f"Number of 9s found and replaced: {num_nines_found}")

# increased_column_10 = df.iloc[:, 10] + 1
# print("Column 10 increased by 1")

# # Update the DataFrame with the modified column
# df.iloc[:, 10] = increased_column_10


num_replaced_zeros = 0
num_replaced_ones = 0

# Identify the last column
last_column = df.columns[-1]

# Extract the values of the last column
values = df[last_column].values

# Find indices of 0s and 1s
indices_zeros = np.where(values == 0)[0]
indices_ones = np.where(values == 1)[0]

# Calculate the number of replacements
num_zeros_to_replace = len(indices_zeros) // 2
num_ones_to_replace = len(indices_ones) // 2

# Select indices to replace (first half of 0s and 1s)
zeros_to_replace = indices_zeros[:num_zeros_to_replace]
ones_to_replace = indices_ones[:num_ones_to_replace]

# Replace 0s with 1s
df.loc[zeros_to_replace, last_column] = 1
num_replaced_zeros = len(zeros_to_replace)

# Replace 1s with 0s
df.loc[ones_to_replace, last_column] = 0
num_replaced_ones = len(ones_to_replace)


df.to_csv('dataset/client_train_data_3p.csv', index=False) 

print("Modifications complete. Saved to 'client_train_data_3poisoned.csv'.")



