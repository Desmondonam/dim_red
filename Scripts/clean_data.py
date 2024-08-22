# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    
    Returns:
    - df (DataFrame): Loaded DataFrame from the CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def check_missing_values(df):
    """
    Check for missing values in each column of the DataFrame.
    
    Parameters:
    - df (DataFrame): Input DataFrame.
    
    Returns:
    - missing_values (DataFrame): DataFrame showing columns with missing values and their counts.
    """
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    if missing_values.empty:
        print("No missing values found.")
    else:
        print("Missing values:")
        print(missing_values)
    return missing_values

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.
    - Numerical columns: Replace missing values with mean.
    - Categorical columns: Replace missing values with mode.
    
    Parameters:
    - df (DataFrame): Input DataFrame.
    
    Returns:
    - df_cleaned (DataFrame): DataFrame with missing values handled.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    mode_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = mode_imputer.fit_transform(df[categorical_cols])
    
    return df

def handle_outliers(df):
    """
    Handle outliers in numerical columns using z-score method (standard deviation).
    
    Parameters:
    - df (DataFrame): Input DataFrame.
    
    Returns:
    - df_cleaned (DataFrame): DataFrame with outliers handled.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    df_cleaned = df[(z_scores < 3).all(axis=1)]  # Adjust the z-score threshold as needed
    
    return df_cleaned

def save_as_csv(df, file_path):
    """
    Save DataFrame to a CSV file.
    
    Parameters:
    - df (DataFrame): DataFrame to be saved.
    - file_path (str): File path to save the CSV file.
    """
    df.to_csv(file_path, index=False)
    print(f"Saved cleaned data to {file_path}")

# Example usage:
if __name__ == "__main__":
    # Load data
    data_path = 'Data/second_hand_cars.csv'
    df = load_data(data_path)
    
    # Check missing values
    check_missing_values(df)
    
    # Handle missing values
    df_cleaned = handle_missing_values(df)
    
    # Handle outliers (optional)
    df_cleaned = handle_outliers(df_cleaned)
    
    # Save cleaned data
    save_path = 'Data/cleaned_data.csv'
    save_as_csv(df_cleaned, save_path)
