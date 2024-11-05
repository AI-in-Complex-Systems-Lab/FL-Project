import pandas as pd

def cut_csv_in_half(input_file, output_file, max_records=50000):
    # Read the CSV file
    try:
        # Load only the first `max_records`
        df = pd.read_csv(input_file, nrows=max_records)
        
        # Write to a new CSV file
        df.to_csv(output_file, index=False)
        print(f'Successfully wrote the first {max_records} records to {output_file}')
    except Exception as e:
        print(f'An error occurred: {e}')

# Usage example
cut_csv_in_half('/Users/mr375368/Desktop/FL-Project/Bekzod/disease_prediction/HeartDisease.csv', 'output.csv')
