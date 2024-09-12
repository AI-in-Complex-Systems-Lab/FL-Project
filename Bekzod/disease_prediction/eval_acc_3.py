import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20, 'font.weight' : 'bold'})

# Load the CSV file from the metrics folder
df = pd.read_csv('/Users/guest1/Documents/GitHub/FL-Project/kenny/working/ids/metrics/70PO_1e/combined_NO/combined_eval_accuracy.csv')  # Replace 'your_file_name.csv' with your actual file name

# Filter rows to only include every 5th round
df_filtered = df[df['round'] % 5 == 0]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df_filtered['round'], df_filtered['3_client_eval_accuracy'], marker='o', label='3 Client', markersize=15, linewidth=5, alpha=0.8)
plt.plot(df_filtered['round'], df_filtered['4_client_eval_accuracy'], marker='o', label='4 Client', markersize=15, linewidth=5, alpha=0.8)
plt.plot(df_filtered['round'], df_filtered['5_client_eval_accuracy'], marker='o', label='5 Client', markersize=15, linewidth=5, alpha=0.8)

# Set plot title and labels
#plt.title('Evaluation Accuracy Per 5 Rounds')
#plt.xlabel('Round')
#plt.ylabel('Accuracy')

# Set x-axis range and ticks with extra space
plt.xticks([0, 5, 10, 15, 20])
plt.xlim(-2, 22)  # Adds padding before 0 and after 20

# Set y-axis range
plt.ylim(0, 0.75)

# Display grid and legend
plt.grid(True)
#plt.legend()

# Save the plot as an image file
plt.savefig('metrics/eval_accuracy_plot.png')  # Saves in the metrics folder as 'eval_accuracy_plot.png'

