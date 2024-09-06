# Intrusion Detection System with Federated Learning using the Flower framework
Example of training of a simple Neural Network model as an Intrusion Detection System for Cybersecurity defense using Federated Learning with the [Flower framework](https://flower.ai/) and the [DNP3 Intrusion Detection Dataset](https://ieee-dataport.org/documents/dnp3-intrusion-detection-dataset).

## Instructions
1. Run the [Flower_IDS_DNP3_preprocessing.ipynb](./Flower_IDS_DNP3_preprocessing.ipynb) notebook for preprocessing the dataset and extracting individual straggler training data and test data.
2. Train and test the model with one of the following 2 options
- Option 1: Use the [docker-compose.yaml](./docker-compose.yaml) to launch an aggretation server and three training stragglers.
- Option 2: Manually launch the aggregation server and then manually launch the three training stragglers.
