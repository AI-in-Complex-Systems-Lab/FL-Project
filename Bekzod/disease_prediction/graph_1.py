import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
plt.rcParams.update({'font.size': 10, 'font.weight' : 'bold'})


# Replace these filenames with your actual filenames
filenames_3_clients = ["/Users/guest2/Desktop/disease_prediction copy 2/metrics/client_1_metrics.csv", "/Users/guest2/Desktop/disease_prediction copy 2/metrics/client_2_metrics.csv", "/Users/guest2/Desktop/disease_prediction copy 2/metrics/client_3_metrics.csv"]
filenames_4_clients = ["/Users/guest2/Desktop/disease_prediction copy 2/metrics/client_1_metrics.csv", "/Users/guest2/Desktop/disease_prediction copy 2/metrics/client_2_metrics.csv", "/Users/guest2/Desktop/disease_prediction copy 2/metrics/client_3_metrics.csv", "/Users/guest2/Desktop/disease_prediction copy 2/metrics/client_4_metrics.csv"]
filenames_5_clients = ["/Users/guest2/Desktop/disease_prediction copy 2/metrics/client_1_metrics.csv", "/Users/guest2/Desktop/disease_prediction copy 2/metrics/client_2_metrics.csv", "/Users/guest2/Desktop/disease_prediction copy 2/metrics/client_3_metrics.csv", "/Users/guest2/Desktop/disease_prediction copy 2/metrics/client_4_metrics.csv", "/Users/guest2/Desktop/disease_prediction copy 2/metrics/client_5_metrics.csv"]
# Read the data into DataFrames
dfs_3_clients = [pd.read_csv(filename) for filename in filenames_3_clients]
dfs_4_clients = [pd.read_csv(filename) for filename in filenames_4_clients]
dfs_5_clients = [pd.read_csv(filename) for filename in filenames_5_clients]

# Initialize the plot
fig, axs = plt.subplots(3, 3, figsize=(18, 15))


# Flatten the axes array for easier iteration
axs = axs.flatten()

# Plot metrics for each client
metrics = ['eval_accuracy', 'eval_loss', 'f1_score']
for i, metric in enumerate(metrics):
    for j, df in enumerate(dfs_3_clients):
        axs[i].plot(df['round'], df[metric], label=f'Client {j+1}', marker = "o", markersize=5, linewidth=2, alpha=0.8)
    
    axs[i].set_xlabel('Round')
    axs[i].set_ylabel(metric.replace('_', ' ').title())
    axs[i].set_title(f'{metric.replace("_", " ").title()} for 3 Clients', fontweight = "bold")
    axs[i].legend()
    axs[i].grid(True)
    axs[i].xaxis.set_major_locator(plt.MultipleLocator(5))


for i, metric in enumerate(metrics):
    for j, df in enumerate(dfs_4_clients):
        axs[i + 3].plot(df['round'], df[metric], label=f'Client {j+1}', marker = "o", markersize=5, linewidth=2, alpha=0.8)
    
    axs[i + 3].set_xlabel('Round')
    axs[i + 3].set_ylabel(metric.replace('_', ' ').title())
    axs[i + 3].set_title(f'{metric.replace("_", " ").title()} for 4 Clients', fontweight = "bold")
    axs[i + 3].legend()
    axs[i + 3].grid(True)
    axs[i + 3].xaxis.set_major_locator(plt.MultipleLocator(5))


for i, metric in enumerate(metrics):
    for j, df in enumerate(dfs_5_clients):
        axs[i + 6].plot(df['round'], df[metric], label=f'Client {j+1}', marker = "o", markersize=5, linewidth=2, alpha=0.8)
    
    axs[i + 6].set_xlabel('Round')
    axs[i + 6].set_ylabel(metric.replace('_', ' ').title())
    axs[i + 6].set_title(f'{metric.replace("_", " ").title()} for 5 Clients', fontweight = "bold")
    axs[i + 6].legend()
    axs[i + 6].grid(True)
    axs[i + 6].xaxis.set_major_locator(plt.MultipleLocator(5))

# Adjust layout
plt.grid(True)
plt.tight_layout()
plt.show()


