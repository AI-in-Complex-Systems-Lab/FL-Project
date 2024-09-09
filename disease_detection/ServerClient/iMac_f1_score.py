import matplotlib.pyplot as plt


cl1 = [0.875070575648278, 0.8750097029461327, 0.8775744219488731, 0.8799909621874714, 0.8806273919644462]
cl2 = 

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(range(1, 6), cl1, label='Client 1', marker="o")

plt.title('F1 Score Over 5 Rounds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 6))

plt.tight_layout()
plt.show()