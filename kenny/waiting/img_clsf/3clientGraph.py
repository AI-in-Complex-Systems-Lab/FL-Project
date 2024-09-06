import pandas as pd
import matplotlib.pyplot as plt

# Extracting every 5 epochs for loss and accuracy
# epochs = list(range(5, 51, 5))
# losses = [1.026323914527893, 1.007663607597351, 1.1107728481292725, 1.2822197675704956, 1.4699701070785522, 
        #   1.7443119287490845, 1.9948707818984985, 2.255345582962036, 2.5626516342163086, 2.753323554992676]
# accuracies = [0.6379, 0.6519, 0.649, 0.6375, 0.627, 0.6198, 0.6121, 0.6117, 0.6012, 0.6047]

epochs = list(range(5, 51, 5))
losses = [1.036600947380066, 1.0043487548828125, 1.1223362684249878, 1.3028602600097656, 1.567495346069336, 1.8765522241592407, 2.154665231704712, 2.507026433944702, 2.764721632003784, 3.0897130966186523]
accuracies = [0.6379, 0.6562, 0.6439, 0.6321, 0.6233999999999998, 0.6156, 0.6078, 0.6064000000000002, 0.6036, 0.6042999999999998]

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

