# Intrusion Detection System with Federated Learning using the Flower framework
Example of training of a simple Neural Network model as an Intrusion Detection System for Cybersecurity defense using Federated Learning with the [Flower framework](https://flower.ai/) and the [DNP3 Intrusion Detection Dataset](https://ieee-dataport.org/documents/dnp3-intrusion-detection-dataset).

## Instructions
1. Run the [Flower_IDS_DNP3_preprocessing.ipynb](./Flower_IDS_DNP3_preprocessing.ipynb) notebook for preprocessing the dataset and extracting individual straggler training data and test data.
2. Train and test the model by manually launch the aggregation server and then manually launch the training stragglers.
