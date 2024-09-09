import matplotlib.pyplot as plt


cl1 = [0.9159162044525146, 0.9160412549972534, 0.9165103435516357, 0.9162601828575134, 0.9159474968910217]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(range(1, 6), cl1, label='Client 1', marker="o")

plt.title('Evaluation Accuracy Over 5 Rounds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 6))

plt.tight_layout()
plt.show()