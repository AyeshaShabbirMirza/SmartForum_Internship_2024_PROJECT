# Import necessary libraries
import pandas as pd

# Load the dataset
data = pd.read_csv(r'C:\Users\Me\Downloads\application_record.csv')

# Display the first few rows to understand the data
print("First few rows of the dataset:\n", data.head())

# Display information about the dataset (data types and non-null counts)
print("\nDataset Info:\n")
print(data.info())

# Check for missing values
print("\nMissing Values in Each Column:\n")
print(data.isnull().sum())

# Describe the dataset to get statistical overview of numerical columns
print("\nStatistical Summary of Numerical Columns:\n")
print(data.describe())

# Check the distribution of unique values in categorical columns
print("\nUnique Values in Categorical Columns:\n")
for col in data.select_dtypes(include=['object']).columns:
    print(f"{col}: {data[col].nunique()} unique values")
    print(data[col].value_counts().head(), "\n")
