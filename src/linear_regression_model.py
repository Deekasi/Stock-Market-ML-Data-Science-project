"""
Linear Regression Module
"""

from sklearn.linear_model import LinearRegression


def train_linear_regression(X_train, y_train):
    """
    Train Linear Regression Model.
    """

    print("=" * 80)
    print("TRAINING LINEAR REGRESSION")
    print("=" * 80)

    model = LinearRegression()

    model.fit(X_train, y_train)

    print("\nModel Training Completed Successfully.")

    return model


def predict_linear_regression(model, X_test):
    """
    Predict using trained model.
    """

    predictions = model.predict(X_test)

    print("\nPrediction Completed Successfully.")

    return predictions