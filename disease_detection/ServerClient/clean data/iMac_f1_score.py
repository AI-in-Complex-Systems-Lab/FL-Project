import matplotlib.pyplot as plt


cl1 = [0.8792689992648858, 0.877590628775184, 0.8812601774987617, 0.8802524870306064, 0.8791752019170368]
cl2 = [0.8763455274981135, 0.8734946384727967, 0.8780630636035411, 0.8773610844711583, 0.8754932678578019]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(range(1, 6), cl1, label='Client 1', marker="o")
plt.plot(range(1, 6), cl2, label='Client 2', marker="o")

plt.title('F1 Score Over 5 Rounds on Clean data')
plt.xlabel('Rounds')
plt.ylabel('F1')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 6))

plt.tight_layout()
plt.show()