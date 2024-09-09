import matplotlib.pyplot as plt


cl1 = [0.9172607660293579, 0.9168855547904968, 0.9176672697067261, 0.9174796938896179, 0.9174171090126038]
cl2 = [0.914853036403656, 0.9141338467597961, 0.9150093793869019, 0.9149468541145325, 0.9146654009819031]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(range(1, 6), cl1, label='Client 1', marker="o")
plt.plot(range(1, 6), cl2, label='Client 1', marker="o")

plt.title('Evaluation Accuracy Over 5 Rounds')
plt.xlabel('Epoch')
plt.ylabel('Eval Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 6))

plt.tight_layout()
plt.show()