import matplotlib.pyplot as plt
import pandas as pd

# Extracting every 5 epochs for loss and accuracy
epochs = list(range(5, 51, 5))
losses = [1.026323914527893, 1.007663607597351, 1.1107728481292725, 1.2822197675704956, 1.4699701070785522, 
          1.7443119287490845, 1.9948707818984985, 2.255345582962036, 2.5626516342163086, 2.753323554992676]
accuracies = [0.6379, 0.6519, 0.649, 0.6375, 0.627, 0.6198, 0.6121, 0.6117, 0.6012, 0.6047]

# Create a DataFrame
data = {
    "Epoch": epochs,
    "Loss": losses,
    "Accuracy": accuracies
}

df = pd.DataFrame(data)

# Plotting the data
plt.figure(figsize=(12, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(df["Epoch"], df["Loss"], label="Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(df["Epoch"], df["Accuracy"], label="Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Epochs")
plt.legend()

plt.tight_layout()
plt.show()