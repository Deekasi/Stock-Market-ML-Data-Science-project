"""
=========================================================
TIME SERIES TRAIN-TEST SPLITTER
=========================================================
Splits stock-market data chronologically.

Important:
    NO SHUFFLING.

Earlier data  -> Training
Later data    -> Testing
=========================================================
"""


def split_data(X, y, train_size=0.80):
    """
    Perform chronological train-test split.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target values.

    train_size : float
        Percentage of data used for training.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    """

    # =================================================
    # CALCULATE SPLIT INDEX
    # =================================================

    split_index = int(
        len(X) * train_size
    )

    # =================================================
    # CHRONOLOGICAL SPLIT
    # =================================================
    #
    # IMPORTANT:
    # We are NOT using random sampling.
    #
    # First 80%  -> Training
    # Last 20%   -> Testing
    # =================================================

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    # =================================================
    # RESET INDEX
    # =================================================

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # =================================================
    # DISPLAY INFORMATION
    # =================================================

    print("\n" + "=" * 80)
    print("TIME-SERIES TRAIN-TEST SPLIT")
    print("=" * 80)

    print("\nTrain size:")
    print(len(X_train))

    print("\nTest size:")
    print(len(X_test))

    print("\nTrain percentage:")
    print(
        round(
            len(X_train) / len(X) * 100,
            2
        ),
        "%"
    )

    print("\nTest percentage:")
    print(
        round(
            len(X_test) / len(X) * 100,
            2
        ),
        "%"
    )

    print("\nShuffling:")
    print("DISABLED")

    print("\nData order:")
    print("Past -> Training | Future -> Testing")

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )