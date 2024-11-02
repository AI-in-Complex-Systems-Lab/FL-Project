import pandas as pd
import os

# Folder containing the CSV files
folder_path = 'metrics'

# Define the filenames and desired new CSV filenames
file_names = ['3_client_global_metrics.csv', '4_client_global_metrics.csv', '5_client_global_metrics.csv']
columns = ['eval_loss', 'eval_accuracy', 'f1_score']
new_file_names = {
    'eval_loss': 'combined_eval_loss.csv',
    'eval_accuracy': 'combined_eval_accuracy.csv',
    'f1_score': 'combined_f1_score.csv'
}

# Initialize empty dataframes to store the combined data
combined_eval_loss = pd.DataFrame()
combined_eval_accuracy = pd.DataFrame()
combined_f1_score = pd.DataFrame()

# Loop through each file and extract the required columns
for i, file_name in enumerate(file_names):
    # Construct the full path to the CSV file
    file_path = os.path.join(folder_path, file_name)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract the round and required metrics, renaming the columns appropriately
    eval_loss = df[['round', 'eval_loss']].rename(columns={'eval_loss': f'{i + 3}_client_eval_loss'})
    eval_accuracy = df[['round', 'eval_accuracy']].rename(columns={'eval_accuracy': f'{i + 3}_client_eval_accuracy'})
    f1_score = df[['round', 'f1_score']].rename(columns={'f1_score': f'{i + 3}_client_f1'})
    
    # Merge with the combined dataframes on the 'round' column
    if combined_eval_loss.empty:
        combined_eval_loss = eval_loss
    else:
        combined_eval_loss = pd.merge(combined_eval_loss, eval_loss, on='round')
        
    if combined_eval_accuracy.empty:
        combined_eval_accuracy = eval_accuracy
    else:
        combined_eval_accuracy = pd.merge(combined_eval_accuracy, eval_accuracy, on='round')
    
    if combined_f1_score.empty:
        combined_f1_score = f1_score
    else:
        combined_f1_score = pd.merge(combined_f1_score, f1_score, on='round')

# Save the combined dataframes to new CSV files
combined_eval_loss.to_csv(os.path.join(folder_path, new_file_names['eval_loss']), index=False)
combined_eval_accuracy.to_csv(os.path.join(folder_path, new_file_names['eval_accuracy']), index=False)
combined_f1_score.to_csv(os.path.join(folder_path, new_file_names['f1_score']), index=False)

print(f"New CSV files created: {new_file_names['eval_loss']}, {new_file_names['eval_accuracy']}, {new_file_names['f1_score']}")
