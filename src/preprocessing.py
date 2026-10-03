"""
Data Preprocessing Module
"""

import os


def clean_dataset(df):
    """
    Clean dataset after feature engineering.
    """

    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)

    print("\nShape Before Cleaning:")
    print(df.shape)

    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())

    # Create target column (next day's Close price)
    df["Target"] = df["Close"].shift(-1)

    # Remove rows with NaN values
    df = df.dropna()

    # Reset index
    df = df.reset_index(drop=True)

    print("\nShape After Cleaning:")
    print(df.shape)

    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())

    return df


def save_processed_data(df, output_path):
    """
    Save processed dataset.
    """

    folder = os.path.dirname(output_path)

    os.makedirs(folder, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nProcessed dataset saved successfully.")
    print(output_path)