import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with the given data
data = {
    # Fig 1
    # "Epoch": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    # "Train Loss": [1.46215, 1.31225, 1.20283, 1.12479, 1.06343, 1.01086, 0.95988, 0.91716, 0.87730, 0.84147],
    # "Train Accuracy": [0.474, 0.530, 0.572, 0.601, 0.626, 0.644, 0.664, 0.678, 0.693, 0.705],
    # "Test Loss": [1.43558, 1.31740, 1.23026, 1.19395, 1.15490, 1.12613, 1.10565, 1.08289, 1.09199, 1.07088],
    # "Test Accuracy": [0.482, 0.527, 0.562, 0.574, 0.594, 0.606, 0.613, 0.621, 0.624, 0.634]data_new = {

    #  Fig 2
    # "Epoch": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    # "Train Loss": [1.43287, 1.29966, 1.20132, 1.11655, 1.05073, 0.99745, 0.95247, 0.91179, 0.87382, 0.83893],
    # "Train Accuracy": [0.487, 0.536, 0.577, 0.606, 0.631, 0.650, 0.667, 0.681, 0.695, 0.706],
    # "Test Loss": [1.42261, 1.31891, 1.23241, 1.16399, 1.13622, 1.11475, 1.09289, 1.08491, 1.07778, 1.07877],
    # "Test Accuracy": [0.484, 0.527, 0.563, 0.590, 0.601, 0.613, 0.622, 0.629, 0.629, 0.628]
}

df = pd.DataFrame(data)

# Plotting the data
plt.figure(figsize=(12, 6))

# Plot Train and Test Loss
plt.subplot(1, 2, 1)
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss", marker='o')
plt.plot(df["Epoch"], df["Test Loss"], label="Test Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train and Test Loss Over Epochs")
plt.legend()

# Plot Train and Test Accuracy
plt.subplot(1, 2, 2)
plt.plot(df["Epoch"], df["Train Accuracy"], label="Train Accuracy", marker='o')
plt.plot(df["Epoch"], df["Test Accuracy"], label="Test Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train and Test Accuracy Over Epochs")
plt.legend()

plt.tight_layout()
plt.show()
