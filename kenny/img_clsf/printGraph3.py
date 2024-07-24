import matplotlib.pyplot as plt

# Define the epochs and accuracies
epochs = list(range(5, 55, 5))  # Epochs from 5 to 50 with a step of 5
ML_accuracies = [0.484, 0.527, 0.563, 0.590, 0.601, 0.613, 0.622, 0.629, 0.629, 0.628]
FL_accuracies = [0.637, 0.656, 0.643, 0.632, 0.623, 0.615, 0.607, 0.606, 0.603, 0.604]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(epochs, ML_accuracies, marker='o', color='blue', label='Machine Learning')
plt.plot(epochs, FL_accuracies, marker='o', color='green', label='Federated Learning')

plt.title('Comparison of Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.xticks(epochs)  # Ensure all epochs are marked
plt.tight_layout()

plt.show()
