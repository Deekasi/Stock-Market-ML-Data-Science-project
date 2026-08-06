import pandas as pd


def inspect_data(df):
    """
    Display basic dataset information.
    """

    print("=" * 50)
    print("Dataset Shape")
    print(df.shape)

    print("\nFirst Five Rows")
    print(df.head())

    print("\nLast Five Rows")
    print(df.tail())

    print("\nColumn Information")
    print(df.info())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())