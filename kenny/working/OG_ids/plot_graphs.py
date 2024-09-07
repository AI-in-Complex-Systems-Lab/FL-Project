import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Define the path to the metrics folder
metrics_dir = "metrics"

# Initialize lists to collect data
rounds = []
train_losses = []
eval_losses = []
train_accuracies = []
eval_accuracies = []
f1_scores = []

# Loop through each CSV file in the metrics folder
for filename in os.listdir(metrics_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(metrics_dir, filename)
        data = pd.read_csv(filepath)
        
        # Append data to lists
        rounds.extend(data['round'])
        train_losses.extend(data['train_loss'])
        eval_losses.extend(data['eval_loss'])
        train_accuracies.extend(data['train_accuracy'])
        eval_accuracies.extend(data['eval_accuracy'])
        f1_scores.extend(data['f1_score'])

# Sort data by rounds
sorted_data = sorted(zip(rounds, train_losses, eval_losses, train_accuracies, eval_accuracies, f1_scores))
rounds, train_losses, eval_losses, train_accuracies, eval_accuracies, f1_scores = zip(*sorted_data)

# Filter data to plot every 5 rounds
rounds_filtered = rounds[::5]
train_losses_filtered = train_losses[::5]
eval_losses_filtered = eval_losses[::5]
train_accuracies_filtered = train_accuracies[::5]
eval_accuracies_filtered = eval_accuracies[::5]
f1_scores_filtered = f1_scores[::5]

# Define x-tick positions and labels
x_ticks = list(range(0, max(rounds_filtered) + 1, 5))

# Plot Train Loss and Eval Loss
plt.figure(figsize=(10, 5))
plt.plot(rounds_filtered, train_losses_filtered, label='Train Loss', marker='o')
plt.plot(rounds_filtered, eval_losses_filtered, label='Eval Loss', marker='x')
plt.xlabel('Rounds')
plt.ylabel('Loss')
plt.title('Train Loss vs Eval Loss (Every 5 Rounds)')
plt.legend()
plt.grid(True)
plt.xticks(x_ticks)  # Set x-axis ticks to specified values
plt.savefig(os.path.join(metrics_dir, 'loss_plot_every_5_rounds.png'))
plt.show()

# Plot Train Accuracy and Eval Accuracy
plt.figure(figsize=(10, 5))
plt.plot(rounds_filtered, train_accuracies_filtered, label='Train Accuracy', marker='o')
plt.plot(rounds_filtered, eval_accuracies_filtered, label='Eval Accuracy', marker='x')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Eval Accuracy (Every 5 Rounds)')
plt.legend()
plt.grid(True)
plt.xticks(x_ticks)  # Set x-axis ticks to specified values
plt.savefig(os.path.join(metrics_dir, 'accuracy_plot_every_5_rounds.png'))
plt.show()

# Plot F1 Score
plt.figure(figsize=(10, 5))
plt.plot(rounds_filtered, f1_scores_filtered, label='F1 Score', marker='s', color='purple')
plt.xlabel('Rounds')
plt.ylabel('F1 Score')
plt.title('F1 Score over Rounds (Every 5 Rounds)')
plt.grid(True)
plt.xticks(x_ticks)  # Set x-axis ticks to specified values
plt.savefig(os.path.join(metrics_dir, 'f1_score_plot_every_5_rounds.png'))
plt.show()
