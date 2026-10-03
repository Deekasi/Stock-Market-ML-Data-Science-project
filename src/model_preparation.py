"""
=========================================================
MODEL PREPARATION
=========================================================
Prepares features (X) and target (y) for machine learning.

Target:
    Tomorrow's closing price.

Example:
    Today's data  --->  Tomorrow's Close
=========================================================
"""

import pandas as pd


def prepare_features_and_target(df):
    """
    Prepare feature matrix X and target y.

    Target:
        Next day's Close price.
    """

    data = df.copy()

    # =================================================
    # CREATE FUTURE TARGET
    # =================================================
    #
    # shift(-1) moves tomorrow's Close to today's row.
    #
    # Example:
    #
    # Date        Close    Target
    # Jan 1       100      105
    # Jan 2       105      102
    # Jan 3       102      108
    #
    # Therefore:
    # Today's features -> Tomorrow's Close
    # =================================================

    data["Target"] = data["Close"].shift(-1)

    # =================================================
    # REMOVE LAST ROW
    # =================================================
    #
    # The final row has no tomorrow's price available.
    #
    # Example:
    # Last Date = July 31
    # Tomorrow's Close = unknown
    #
    # Therefore remove it.
    # =================================================

    data = data.dropna(subset=["Target"])

    # =================================================
    # REMOVE COLUMNS NOT USED AS FEATURES
    # =================================================

    columns_to_remove = [
        "Target",
        "Date",
    ]

    X = data.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    y = data["Target"]

    # =================================================
    # KEEP ONLY NUMERIC FEATURES
    # =================================================

    X = X.select_dtypes(
        include=["number"]
    )

    # =================================================
    # REMOVE INFINITE VALUES
    # =================================================

    X = X.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    # =================================================
    # REMOVE ROWS WITH MISSING FEATURES
    # =================================================

    valid_rows = X.notna().all(axis=1)

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # =================================================
    # RESET INDEX
    # =================================================

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # =================================================
    # DISPLAY INFORMATION
    # =================================================

    print("\n" + "=" * 80)
    print("MODEL PREPARATION")
    print("=" * 80)

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nNumber of features:")
    print(X.shape[1])

    print("\nNumber of samples:")
    print(X.shape[0])

    print("\nX shape:")
    print(X.shape)

    print("\ny shape:")
    print(y.shape)

    print("\nTarget:")
    print("Next day's Close price")

    print("\nFirst 5 target values:")
    print(y.head())

    return X, y