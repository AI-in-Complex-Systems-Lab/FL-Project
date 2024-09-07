# centralized_ml.py
import argparse
import os
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# Define the base directory as the current directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to ids_dnp3 directory
ids_dnp3_datasets_path = os.path.join(base_dir, 'datasets', 'federated_datasets')

def get_model():
      return model

def get_X_train_scaled():
     return X_train_scaled

def get_X_test_scaled():
     return X_test_scaled

def get_y_train_cat():
     return y_train_cat

def get_y_test_cat():
     return y_test_cat

def get_y_test():
     return y_test

def get_early_stop():
     return early_stop
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Centralized Machine Learning Implementation')
    parser.add_argument("-d", "--dataset", help="Dataset directory", default=ids_dnp3_datasets_path)
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        sys.exit(f"Wrong path to directory with datasets: {args.dataset}")

    # Load train and test data
    df_train = pd.read_csv(os.path.join(args.dataset, 'train_data.csv'))
    df_test = pd.read_csv(os.path.join(args.dataset, 'test_data.csv'))

    # Split data into X and y
    X_train = df_train.drop(columns=['y']).to_numpy()
    y_train = df_train['y'].to_numpy()
    X_test = df_test.drop(columns=['y']).to_numpy()
    y_test = df_test['y'].to_numpy()

    # Scale feature values for input data normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use one-hot-vectors for label representation
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    # Define a MLP model
    model = Sequential([
        InputLayer(input_shape=(X_train_scaled.shape[1],)),
        Dense(units=50, activation='relu'),
        Dropout(0.2),
        Dense(units=y_train_cat.shape[1], activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # Train the model
    model.fit(
        X_train_scaled, y_train_cat,
        epochs=100,
        validation_data=(X_test_scaled, y_test_cat),
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=1)
    f1 = f1_score(y_test, np.argmax(model.predict(X_test_scaled), axis=1), average='weighted')

    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
