import errno
import os
import py7zr
import re
import shutil
import sys
import time
import urllib
import collections
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

pd.set_option('future.no_silent_downcasting', True)

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def download(url, filename):
    urllib.request.urlretrieve(url, filename, reporthook)

def extract(archive, target_dir):
    archive = py7zr.SevenZipFile(archive, mode='r')
    try:
        os.makedirs(target_dir)
        archive.extractall(path=target_dir)
        archive.close()
        print(f"Archive '{archive}' extracted successfully!")
    except OSError as e:
        if e.errno == errno.EEXIST:
            archive.extractall(path=target_dir)
            archive.close()
            print(f"Archive '{archive}' extracted successfully!")
        else:
            print(f"Archive '{archive}' failed to extract...")

base_data_dir = './datasets'

if not os.path.isdir(os.path.join(base_data_dir, 'Training_Testing_Balanced_CSV_Files')):
    filename = 'DNP3_Intrusion_Detection_Dataset_Final.7z'
    
    if not py7zr.is_7zfile(filename):
        url = ('https://zenodo.org/records/7348493/files/DNP3_Intrusion_Detection_Dataset_Final.7z?download=1')
        download(url, filename)
    
    extract(filename, base_data_dir)

    innerDirs = os.listdir(base_data_dir)
    for dir in innerDirs:
        if re.search('2020*', dir):
            shutil.rmtree(os.path.join(base_data_dir, dir))
    
    os.remove(os.path.join('.', filename))

tp = 'cic'      # Type: dataset type (CICFlowMeter or Custom Python parser), choose between 'cic' and 'custom'
n_workers = 5   # N: number of federated workers

assert tp in ('cic', 'custom'), "Wrong dataset type, choose between 'cic' and 'custom'"
assert n_workers == 5, "This script is set for 5 workers"

path = os.path.join(base_data_dir, 'Training_Testing_Balanced_CSV_Files')

if tp == 'cic':
    dataset = f'CICFlowMeter'
else:
    dataset = 'Custom_DNP3_Parser'

if 'CIC' in dataset:
    train_csv = os.path.join(path, dataset, 'CICFlowMeter_Training_Balanced.csv')
    test_csv = os.path.join(path, dataset, 'CICFlowMeter_Testing_Balanced.csv')
elif 'Custom' in dataset:
    train_csv = os.path.join(path, dataset, 'Custom_DNP3_Parser_Training_Balanced.csv')
    test_csv = os.path.join(path, dataset, 'Custom_DNP3_Parser_Testing_Balanced.csv')
else:
    raise Exception("Wrong dataset")

df_train = pd.read_csv(train_csv, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
df_test = pd.read_csv(test_csv, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

df_train['Label'] = df_train['Label'].str.lower()
df_test['Label'] = df_test['Label'].str.lower()

unique_labels = list(df_train.Label.astype('category').unique())
unique_codes = list(df_train.Label.astype('category').cat.codes.unique())

mapping = dict(zip(unique_labels, unique_codes))
mapping_inv = dict(zip(unique_codes, unique_labels))

df_train['Label'] = df_train['Label'].astype('category').cat.rename_categories(mapping)
df_test['Label'] = df_test['Label'].astype('category').cat.rename_categories(mapping)

train = df_train.iloc[:, 9:]
test = df_test.iloc[:, 9:]

train.rename(columns={"Label": "y"}, inplace=True, errors="raise")
test.rename(columns={"Label": "y"}, inplace=True, errors="raise")

train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

train.dropna(inplace=True)
test.dropna(inplace=True)

directory = os.path.join(base_data_dir, 'federated_datasets')

try:
    os.makedirs(directory)
    train.to_csv(os.path.join(directory, 'train_data.csv'), index=False)
    test.to_csv(os.path.join(directory, 'test_data.csv'), index=False)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
    else:   
        train.to_csv(os.path.join(directory, 'train_data.csv'), index=False)
        test.to_csv(os.path.join(directory, 'test_data.csv'), index=False)

# Define proportions for each worker
proportions = [0.10, 0.20, 0.27, 0.13, 0.30]
total_samples = len(train)  # Use original dataset size
sample_counts = [int(total_samples * prop) for prop in proportions]

client_data = []
train_copy = train.copy()

for i, n_samples in enumerate(sample_counts):
    # Sample without dropping rows yet
    sample = train_copy.sample(n=n_samples, random_state=i)
    client_data.append(sample)

# Drop all sampled rows from train_copy after collecting all samples
sampled_indices = pd.concat(client_data).index
train_copy.drop(index=sampled_indices, inplace=True)

# Save each client's data to a separate file
for i, sample in enumerate(client_data):
    sample.to_csv(os.path.join(directory, f'client_train_data_{i+1}.csv'), index=False)
    print(f"Worker {i+1} training data contains {len(sample)} points")

# Plot label counts for each worker's dataset
fig = plt.figure(figsize=(20, 7))
fig.suptitle('Label Counts for Each Worker\'s Dataset')
fig.tight_layout()

for i in range(n_workers):
    plot_data = collections.defaultdict(list)
    sample = client_data[i]
    
    for label in sample['y']:
        plot_data[label].append(label)
    
    n_cols = n_workers if n_workers < 5 else 5
    xlim = [0, max(len(v) for v in plot_data.values()) + 10]
    ylim = [min(unique_codes)-1, max(unique_codes)+1]
    yticks = list(range(min(unique_codes), max(unique_codes)+1))
    yticks_labels = [mapping_inv[k] for k in range(0, max(unique_codes)+1)]
    
    plt.subplot(int(n_workers / 5)+1, n_cols, i+1)
    plt.subplots_adjust(wspace=0.3)
    plt.title(f'Worker {i+1}')
    plt.xlabel('# points')
    plt.xlim(xlim)
    plt.ylabel('Label')
    plt.ylim(ylim)
    plt.yticks(yticks, labels=yticks_labels)

    # Plot values on top of bars
    for key in plot_data:
        if len(plot_data[key]) > 0:
            plt.text(len(plot_data[key])+8, int(key)-0.1, str(len(plot_data[key])), ha='center')
    
    for j in range(min(unique_codes), max(unique_codes)+1):
        plt.hist(
            plot_data[j],
            density=False,
            bins=[k-0.5 for k in range(min(unique_codes), max(unique_codes)+2)],
            orientation='horizontal'
        )

# plt.show()  # Uncomment to display the plot
