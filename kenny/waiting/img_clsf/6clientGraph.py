import matplotlib.pyplot as plt

# Extracted every 5 epochs for loss and accuracy
epochs = list(range(5, 55, 5))
losses = [
    1.4073359966278076, 1.2708178758621216, 1.196216344833374, 1.1488018035888672,
    1.1177209615707397, 1.0970114469528198, 1.083072304725647, 1.0759284496307373, 
    1.0733875036239624, 1.0775887966156006
]
accuracies = [
    0.4902, 0.5429, 0.5761, 0.5969, 0.6048, 0.6159, 0.6235, 0.6283, 0.6291, 0.6319
]

# Plotting the loss over epochs
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, losses, marker='o', color='red', label='Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.xticks(epochs)

# Plotting the accuracy over epochs
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracies, marker='o', color='blue', label='Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.xticks(epochs)

plt.tight_layout()
plt.show()
