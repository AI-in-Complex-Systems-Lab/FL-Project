import matplotlib.pyplot as plt


cl1_poisoned = [0.8779832658137171, 0.8788681154806739, 0.8830802756911283, 0.879217551522454, 0.8786050334319938]
cl2_poisoned = [0.8738564282029928, 0.8754184214370694, 0.8795246098474043, 0.875875253609748, 0.8745543472909894]


cl1_clean = [0.8792689992648858, 0.877590628775184, 0.8812601774987617, 0.8802524870306064, 0.8791752019170368]
cl2_clean = [0.8763455274981135, 0.8734946384727967, 0.8780630636035411, 0.8773610844711583, 0.8754932678578019]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(range(1, 6), cl1_poisoned, label='Client 1 Poisoned data', marker="o")
plt.plot(range(1, 6), cl2_poisoned, label='Client 2 Poisoned data', marker="o")

plt.plot(range(1, 6), cl1_clean, label='Client 1 Clean data', marker="o")
plt.plot(range(1, 6), cl2_clean, label='Client 2 Clean data', marker="o")

plt.title('F1 Score Over 5 Rounds on Poisoned and Clean data')
plt.xlabel('Rounds')
plt.ylabel('F1')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 6))

plt.tight_layout()
plt.show()