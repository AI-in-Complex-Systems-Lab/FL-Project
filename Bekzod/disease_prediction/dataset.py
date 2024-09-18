import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import sys

def load_heart_disease_data():
    # Ensure the dataset directory exists
    os.makedirs('dataset', exist_ok=True)
    
    # Import the dataset from the 'dataset' folder
    df = pd.read_csv('/home/jetbot/Desktop/FL-Project/Bekzod/disease_prediction/HeartDisease.csv')

    # Split data into features (X) and labels (y)
    X = df.drop(['HeartDisease'], axis=1)
    y = df['HeartDisease']

    #print()

    # Check for NaN values in X
    nan_rows = X[X.isnull().any(axis=1)]

    # Print rows with NaN values
    #print("Rows with NaN values:")
    #print(nan_rows)

    # Remove rows with NaN values from X
    X = X.dropna().reset_index(drop=True)

    # Drop corresponding y values which had NaN in X
    y = y.drop(nan_rows.index).reset_index(drop=True)

    # Encoding the independent variables
    columns_to_encode = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 
                         'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 
                         'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']
    
    # Create the ColumnTransformer
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), columns_to_encode)], remainder='passthrough')
    # Apply the transformation to the selected columns
    X = np.array(ct.fit_transform(X))

    # Encoding the dependent variable
    y = y.map({'No': 0, 'Yes': 1})
    y = to_categorical(y, num_classes=2)

    # Standardize the feature variables
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Split the data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the splits to CSV files in the dataset folder
    train_data = pd.DataFrame(X_train)
    train_data['HeartDisease'] = np.argmax(y_train, axis=1)  # Add back the encoded labels
    train_data.to_csv('dataset/train_data.csv', index=False)

    test_data = pd.DataFrame(X_test)
    test_data['HeartDisease'] = np.argmax(y_test, axis=1)  # Add back the encoded labels
    test_data.to_csv('dataset/test_data.csv', index=False)

    return X_train, X_test, y_train, y_test


def split_train_data():
    # Load the training data
    train_data_path = 'dataset/train_data.csv'
    
    # Check if the file exists
    if not os.path.exists(train_data_path):
        print(f"Error: {train_data_path} not found.")
        sys.exit(1)
    
    # Read the training data
    df = pd.read_csv(train_data_path)
    
    # Split the data into three parts
    num_rows = len(df)
    split_size = num_rows // 5

    # Split the data
    df_1 = df.iloc[:split_size]
    df_2 = df.iloc[split_size:2*split_size]
    df_3 = df.iloc[2*split_size:3*split_size]
    df_4 = df.iloc[3*split_size:]
    df_5 = df.iloc[4*split_size:]

    # Save the splits to CSV files
    df_1.to_csv('dataset/client_train_data_1.csv', index=False)
    df_2.to_csv('dataset/client_train_data_2.csv', index=False)
    df_3.to_csv('dataset/client_train_data_3.csv', index=False)
    df_4.to_csv('dataset/client_train_data_4.csv', index=False)
    df_5.to_csv('dataset/client_train_data_5.csv', index=False)

# Perform the split
split_train_data()

# Run the function
X_train, X_test, y_train, y_test = load_heart_disease_data()
