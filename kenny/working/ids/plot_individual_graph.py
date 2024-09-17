import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the metrics folder
metrics_dir = "metrics"

# Create the metrics directory if it doesn't exist
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)

# List to hold client files
client_files = [f for f in os.listdir(metrics_dir) if f.endswith(".csv") and 'client_' in f]

# Define x-tick positions and labels for every 5 rounds from 0 to 20
x_ticks = list(range(0, 21, 5))

for file in client_files:
    client_id = file.split('_')[1]  # Extract client ID from the filename
    filepath = os.path.join(metrics_dir, file)
    
    # Read data
    data = pd.read_csv(filepath)
    
    # Filter data to include only every 5th round from 0 to 20
    data_filtered = data[data['round'].isin(x_ticks)]
    
    # Extract filtered data
    rounds = data_filtered['round']
    train_losses = data_filtered['train_loss']
    eval_losses = data_filtered['eval_loss']
    train_accuracies = data_filtered['train_accuracy']
    eval_accuracies = data_filtered['eval_accuracy']
    f1_scores = data_filtered['f1_score']
    
    # Plot Train Loss and Eval Loss
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, train_losses, label='Train Loss', marker='o')
    plt.plot(rounds, eval_losses, label='Eval Loss', marker='x')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title(f'Client {client_id} - Train Loss vs Eval Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(x_ticks)  # Set x-axis ticks to specified values
    plt.savefig(os.path.join(metrics_dir, f'client_{client_id}_loss_plot.png'))
    plt.close()

    # Plot Train Accuracy and Eval Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(rounds, eval_accuracies, label='Eval Accuracy', marker='x')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title(f'Client {client_id} - Train Accuracy vs Eval Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(x_ticks)  # Set x-axis ticks to specified values
    plt.savefig(os.path.join(metrics_dir, f'client_{client_id}_accuracy_plot.png'))
    plt.close()
