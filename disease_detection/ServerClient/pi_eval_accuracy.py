import matplotlib.pyplot as plt


cl1 = [0.9171982407569885, 0.9172295331954956, 0.917292058467865, 0.9173858761787415, 0.917854905128479]
cl2 = [0.9145403504371643, 0.9146654009819031, 0.9146341681480408, 0.9147592186927795, 0.9151657223701477]

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