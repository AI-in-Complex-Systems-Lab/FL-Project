import matplotlib.pyplot as plt


cl1 = [0.878648142102272, 0.8791356365844576, 0.8788742927782817, 0.8792761735835624, 0.8823053286414877]
cl2 = [0.8749582974032772, 0.8759000081635407, 0.8750660743013637, 0.8758338033778189, 0.8790778753650902]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(range(1, 6), cl1, label='Client 1', marker="o")
plt.plot(range(1, 6), cl2, label='Client 2', marker="o")

plt.title('F-1 score Over 5 Rounds')
plt.xlabel('Epoch')
plt.ylabel('F-1 score')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 6))

plt.tight_layout()
plt.show()