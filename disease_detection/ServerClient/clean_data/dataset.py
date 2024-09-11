# dataset

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

def load_heart_disease_data():
    
    # import the dataset
    df = pd.read_csv('HeartDisease.csv')

    # Split data into features (X) and labels (y)
    X = df.drop(['HeartDisease'], axis=1)
    y = df['HeartDisease']
    
    print(X)
    
    print(y)
    
    # Check for NaN values in X
    nan_rows = X[X.isnull().any(axis=1)]

    # Print rows with NaN values
    print("Rows with NaN values:")
    print(nan_rows)

    # Remove rows with NaN values from X
    X = X.dropna()

    # Reset index after dropping rows
    X =X.reset_index(drop=True)
    
    #drop corresponding y values which had nan in X
    y = y.drop(nan_rows.index)
    y = y.reset_index(drop=True)

    # Encoding the independent variable
    columns_to_encode = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth','Asthma', 'KidneyDisease', 'SkinCancer']
    # Create the ColumnTransformer
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), columns_to_encode)], remainder='passthrough')
    # Apply the transformation to the selected columns
    X = np.array(ct.fit_transform(X))
    
    # Encoding the dependent Variable
    y = y.map({'No': 0, 'Yes': 1})
    y = to_categorical(y, num_classes=2)
    
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    return X, y
