import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with the given data
data = {
    # Fig 1
    # "Epoch": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    # "Train Loss": [1.46215, 1.31225, 1.20283, 1.12479, 1.06343, 1.01086, 0.95988, 0.91716, 0.87730, 0.84147],
    # "Train Accuracy": [0.474, 0.530, 0.572, 0.601, 0.626, 0.644, 0.664, 0.678, 0.693, 0.705],
    # "Test Loss": [1.43558, 1.31740, 1.23026, 1.19395, 1.15490, 1.12613, 1.10565, 1.08289, 1.09199, 1.07088],
    # "Test Accuracy": [0.482, 0.527, 0.562, 0.574, 0.594, 0.606, 0.613, 0.621, 0.624, 0.634]

    #  Fig 2
    # "Epoch": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    # "Train Loss": [1.43287, 1.29966, 1.20132, 1.11655, 1.05073, 0.99745, 0.95247, 0.91179, 0.87382, 0.83893],
    # "Train Accuracy": [0.487, 0.536, 0.577, 0.606, 0.631, 0.650, 0.667, 0.681, 0.695, 0.706],
    # "Test Loss": [1.42261, 1.31891, 1.23241, 1.16399, 1.13622, 1.11475, 1.09289, 1.08491, 1.07778, 1.07877],
    # "Test Accuracy": [0.484, 0.527, 0.563, 0.590, 0.601, 0.613, 0.622, 0.629, 0.629, 0.628]

    # "Epoch": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    # "Train Loss": [1.43633, 1.29252, 1.20283, 1.13262, 1.07623, 1.02497, 0.97694, 0.93288, 0.89482, 0.85701],
    # "Train Accuracy": [0.483, 0.540, 0.573, 0.600, 0.620, 0.640, 0.657, 0.672, 0.686, 0.701],
    # "Test Loss": [1.40690, 1.29435, 1.23333, 1.19731, 1.16050, 1.14559, 1.12279, 1.11137, 1.09934, 1.09972],
    # "Test Accuracy": [0.497, 0.537, 0.562, 0.578, 0.591, 0.599, 0.608, 0.615, 0.621, 0.626]

    #  Fig 3
    # "Epoch": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    # "Train Loss": [1.43540, 1.26924, 1.16426, 1.09087, 1.03652, 0.98956, 0.94738, 0.90877, 0.87213, 0.83774],
    # "Train Accuracy": [0.481, 0.548, 0.590, 0.617, 0.636, 0.655, 0.669, 0.682, 0.697, 0.708],
    # "Test Loss": [1.41982, 1.27485, 1.20233, 1.15038, 1.11985, 1.08083, 1.07028, 1.05845, 1.06394, 1.04065],
    # "Test Accuracy": [0.489, 0.546, 0.571, 0.597, 0.608, 0.620, 0.625, 0.631, 0.632, 0.643]

    "Epoch": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    "Train Loss": [1.47700, 1.30453, 1.20332, 1.12965, 1.07132, 1.02126, 0.97905, 0.93938, 0.90035, 0.86661],
    "Train Accuracy": [0.468, 0.534, 0.573, 0.601, 0.621, 0.640, 0.654, 0.670, 0.683, 0.694],
    "Test Loss": [1.45032, 1.30196, 1.22507, 1.17433, 1.15130, 1.11413, 1.09523, 1.08509, 1.06995, 1.07035],
    "Test Accuracy": [0.479, 0.533, 0.564, 0.580, 0.595, 0.611, 0.614, 0.621, 0.633, 0.628]
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
