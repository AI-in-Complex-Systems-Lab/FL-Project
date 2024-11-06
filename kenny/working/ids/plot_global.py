import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams.update({'font.size': 20, 'font.weight' : 'bold'})

metrics_dir = "metrics"

# Define the specific files to look for
expected_files = [
    "3_client_global_metrics.csv",
    "4_client_global_metrics.csv",
    "5_client_global_metrics.csv"
]

available_files = [file for file in expected_files if os.path.isfile(os.path.join(metrics_dir, file))]

if not available_files:
    print("No matching metrics files found in the directory.")
else:
    # Loop through each available file and process it
    for metrics_file in available_files:
        # Construct the full path to the metrics file
        metrics_path = os.path.join(metrics_dir, metrics_file)
        
        # Read the metrics file into a DataFrame
        df = pd.read_csv(metrics_path)

# Filter rows for every 5 rounds
df_filtered = df[df["round"] % 5 == 0]

# Plot the metrics
plt.figure(figsize=(12, 6), dpi= 600)

# Plot eval_loss
plt.plot(df_filtered["round"], df_filtered["eval_loss"], marker='o', markersize=10, label='Evaluation Loss', color='red', linewidth=3, alpha=0.9)
#plt.plot(df_filtered["round"], df_filtered["eval_loss"], marker='o', markersize=10, color='red', linewidth=3, alpha=0.9)

# Plot eval_accuracy
plt.plot(df_filtered["round"], df_filtered["eval_accuracy"], marker='o', markersize=10, label='Evaluation Accuracy', color='blue', linewidth=3, alpha=0.9)
#plt.plot(df_filtered["round"], df_filtered["eval_accuracy"], marker='o', markersize=10, color='blue', linewidth=3, alpha=0.9)

# Plot f1_score
plt.plot(df_filtered["round"], df_filtered["f1_score"], marker='o', markersize=10, label='F1 Score', color='green', linewidth=3, alpha=0.9)
#plt.plot(df_filtered["round"], df_filtered["f1_score"], marker='o', markersize=10, color='green', linewidth=3, alpha=0.9)

# Add labels and title
plt.xlabel('Round')
plt.ylabel('Metric Value')
plt.title('Federated Learning Metrics Over Rounds')
plt.legend()
plt.grid(True)
plt.xticks(df_filtered["round"])  # Ensure all round numbers are shown on x-axis

# Save the plot
plt.savefig("metrics/global_metrics_plot.png")
