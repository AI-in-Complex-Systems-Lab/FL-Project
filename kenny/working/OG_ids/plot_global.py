import matplotlib.pyplot as plt
import pandas as pd
import os

# Define the path to the metrics file
metrics_file = os.path.join("metrics", "global_metrics.csv")

# Read the metrics file into a DataFrame
df = pd.read_csv(metrics_file)

# Filter rows for every 5 rounds
df_filtered = df[df["round"] % 5 == 0]

# Plot the metrics
plt.figure(figsize=(12, 6))

# Plot eval_loss
plt.plot(df_filtered["round"], df_filtered["eval_loss"], marker='o', label='Evaluation Loss', color='red')

# Plot eval_accuracy
plt.plot(df_filtered["round"], df_filtered["eval_accuracy"], marker='o', label='Evaluation Accuracy', color='blue')

# Plot f1_score
plt.plot(df_filtered["round"], df_filtered["f1_score"], marker='o', label='F1 Score', color='green')

# Add labels and title
plt.xlabel('Round')
plt.ylabel('Metric Value')
plt.title('Federated Learning Metrics Over Rounds')
plt.legend()
plt.grid(True)
plt.xticks(df_filtered["round"])  # Ensure all round numbers are shown on x-axis

# Save the plot
plt.savefig("metrics/global_metrics_plot.png")
