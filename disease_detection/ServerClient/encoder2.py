from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Copy the dataset to keep original data intact

file_path = "HeartDisease_Outliers.csv"  
data = pd.read_csv(file_path)
encoded_data = data.copy()

# List of categorical columns to encode
categorical_columns = [
    "HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", 
    "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", 
    "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"
]

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode each categorical column
for col in categorical_columns:
    encoded_data[col] = label_encoder.fit_transform(encoded_data[col])
encoded_file_path = "Encoded_HeartDisease_Outliers.csv"  # Specify the output file name
encoded_data.to_csv(encoded_file_path, index=False)
# Display the first few rows of the encoded dataset
encoded_data.head()
