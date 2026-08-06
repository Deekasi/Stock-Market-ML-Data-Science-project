"""
EDA Module
"""

import pandas as pd


def load_data(file_path):
    """
    Load and clean the stock dataset.
    """

    df = pd.read_csv(file_path)

    # Rename first column
    df.rename(columns={"Price": "Date"}, inplace=True)

    # Remove unwanted rows
    df = df.iloc[2:].reset_index(drop=True)

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])

    # Convert remaining columns to numeric
    numeric_columns = ["Close", "High", "Low", "Open", "Volume"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])

    return df


def basic_information(df):
    """
    Display basic information.
    """

    print("=" * 60)
    print("Shape")
    print(df.shape)

    print("=" * 60)
    print("Columns")
    print(df.columns)

    print("=" * 60)
    print("Data Types")
    print(df.dtypes)

    print("=" * 60)
    print("First Five Rows")
    print(df.head())

    print("=" * 60)
    print("Last Five Rows")
    print(df.tail())


def check_missing_values(df):
    """
    Check missing values.
    """

    print("=" * 60)
    print("Missing Values")
    print("=" * 60)

    print(df.isnull().sum())

    print("\nTotal Missing Values:", df.isnull().sum().sum())


def check_duplicates(df):
    """
    Check duplicate rows.
    """

    print("=" * 60)
    print("Duplicate Rows")
    print("=" * 60)

    print("Total Duplicate Rows:", df.duplicated().sum())


def summary_statistics(df):
    """
    Summary statistics.
    """

    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    print(df.describe())