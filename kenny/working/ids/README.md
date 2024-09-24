# Intrusion Detection System with Federated Learning using the Flower framework
This is an example of training of a simple Neural Network model as an Intrusion Detection System for Cybersecurity defense using Federated Learning with the [Flower framework](https://flower.ai/) and the [DNP3 Intrusion Detection Dataset](https://ieee-dataport.org/documents/dnp3-intrusion-detection-dataset).

## Instructions
If you do not see Training_Testing_Balanced_CSV_Files folder, start with step 1, if not, start with step 2:
1. Run the command "python3 preprocess.py" for preprocessing the dataset and extracting individual straggler training data and test data.
2. Train and test the model by manually launch the aggregation server, for example, to run training with three clients, you would run "3_clients_server.py"
3. After the server is initialzed, you will open a new terminal and type "python3 client.py --id 1" then you will increase the number at the end to the number of clients you would like to run the test on.
4. After the training, you can run "python3 plot_combine_graph.py" to see three graphs, one comparison between training and evaluation loss and one for training and evluation accuracy, and a graph of f1-score.
5. You can also run "python3 plot_global.py" for a graph of global model's metries.

This will conclude the the whole training process. For data poisoning, after we have preprocessed the datasets, you can run "python3 data_poison_client3.py", then you will need to manually change the name of the orginal client 3 training data into "client_train_data_3n.csv", then you will need to change the poisoned data csv from "client_train_data_3p.csv" to "client_train_data_3.csv". 

After this, you can continue with step two above and finish the training. 
