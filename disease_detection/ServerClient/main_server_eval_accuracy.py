import matplotlib.pyplot as plt


cl1 = [0.9173858761787415, 0.9175735116004944, 0.9173233509063721, 0.9168230295181274, 0.917854905128479]
cl2 = [0.9146341681480408, 0.914853036403656, 0.9148217439651489, 0.9141025543212891, 0.915603518486023]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(range(1, 6), cl1, label='Client 1', marker="o")
plt.plot(range(1, 6), cl2, label='Client 2', marker="o")

plt.title('Evaluation Accuracy Over 5 Rounds')
plt.xlabel('Epoch')
plt.ylabel('Eval Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 6))

plt.tight_layout()
plt.show()