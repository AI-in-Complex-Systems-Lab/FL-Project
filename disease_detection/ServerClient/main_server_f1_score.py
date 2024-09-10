import matplotlib.pyplot as plt


cl1 = [0.8790411530592084, 0.8804171977945684, 0.8793018530832527, 0.8774380499339411, 0.8840174170417974]
cl2 = [0.8752426349045551, 0.8770287618286212, 0.8759828577892421, 0.8733575857943607, 0.8812788590779911]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(range(1, 6), cl1, label='Client 1', marker="o")
plt.plot(range(1, 6), cl2, label='Client 1', marker="o")

plt.title('F1 Score Over 5 Rounds')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 6))

plt.tight_layout()
plt.show()