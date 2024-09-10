import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20, 'font.weight' : 'bold'})

# Define the path to the metrics folder
metrics_dir = "metrics"

# Create the metrics directory if it doesn't exist
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)

# List to hold client files
client_files = [f for f in os.listdir(metrics_dir) if f.endswith(".csv") and 'client_' in f]

# Initialize dictionaries to hold metrics data
all_data = {
    'train_loss': {},
    'train_accuracy': {},
    'eval_loss': {},
    'eval_accuracy': {},
    'f1_score': {}
}
all_rounds = []

# Extract and sort client IDs
client_ids = sorted([int(file.split('_')[1].split('.')[0]) for file in client_files])

# Iterate over each client file
for client_id in client_ids:
    file = f"client_{client_id}_metrics.csv"
    filepath = os.path.join(metrics_dir, file)
    
    # Read data
    data = pd.read_csv(filepath)
    
    # Extract data for plotting
    rounds = data['round']
    train_loss = data['train_loss']
    train_accuracy = data['train_accuracy']
    eval_loss = data['eval_loss']
    eval_accuracy = data['eval_accuracy']
    f1_scores = data['f1_score']
    
    # Filter data to plot every 5 rounds, ensuring the last round is included if it's not a multiple of 5
    rounds_filtered = [r for r in rounds if r % 5 == 0 or r == max(rounds)]
    train_loss_filtered = [train_loss[i] for i in range(len(rounds)) if rounds[i] in rounds_filtered]
    train_accuracy_filtered = [train_accuracy[i] for i in range(len(rounds)) if rounds[i] in rounds_filtered]
    eval_loss_filtered = [eval_loss[i] for i in range(len(rounds)) if rounds[i] in rounds_filtered]
    eval_accuracy_filtered = [eval_accuracy[i] for i in range(len(rounds)) if rounds[i] in rounds_filtered]
    f1_scores_filtered = [f1_scores[i] for i in range(len(rounds)) if rounds[i] in rounds_filtered]
    
    # Store metrics for each client
    all_rounds = rounds_filtered
    all_data['train_loss'][f'Client {client_id}'] = train_loss_filtered
    all_data['train_accuracy'][f'Client {client_id}'] = train_accuracy_filtered
    all_data['eval_loss'][f'Client {client_id}'] = eval_loss_filtered
    all_data['eval_accuracy'][f'Client {client_id}'] = eval_accuracy_filtered
    all_data['f1_score'][f'Client {client_id}'] = f1_scores_filtered

# Helper function to plot metrics
def plot_metrics(metric_name, ylabel, y_limits=None):
    plt.figure(figsize=(12, 6), dpi= 600)
    for client_id in sorted(client_ids):
        client_label = f'Client {client_id}'
        if client_label in all_data[metric_name]:
            plt.plot(all_rounds, all_data[metric_name][client_label], label=client_label, marker='o', markersize=10, linewidth=3, alpha=0.9)
    #plt.xlabel('Rounds')
    #plt.ylabel(ylabel)
    #plt.title(f'{ylabel} of All Clients over Rounds')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, max(all_rounds) + 1, 5))  # Set x-axis ticks to specified values
    if y_limits:
        plt.ylim(y_limits)  # Set y-axis limits if specified
    plt.savefig(os.path.join(metrics_dir, f'all_clients_{metric_name}_plot.png'))
    plt.close()

# Plot all metrics with specified y-axis limits
plot_metrics('train_loss', 'Train Loss', y_limits=(0.6, 0.8))
plot_metrics('train_accuracy', 'Train Accuracy', y_limits=(0, 1))
plot_metrics('eval_loss', 'Evaluation Loss', y_limits=(0.475, 0.6))
plot_metrics('eval_accuracy', 'Evaluation Accuracy', y_limits=(0.6, 0.8))
plot_metrics('f1_score', 'F1 Score', y_limits=(0.6, 0.8))

