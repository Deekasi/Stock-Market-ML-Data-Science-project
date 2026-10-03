"""
=========================================================
Random Forest Regression Model
=========================================================
"""

from sklearn.ensemble import RandomForestRegressor


def train_random_forest(X_train, y_train):
    """
    Train Random Forest Regressor
    """

    print("=" * 80)
    print("TRAINING RANDOM FOREST")
    print("=" * 80)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("\nRandom Forest Training Completed.")

    return model


def predict_random_forest(model, X_test):
    """
    Make predictions using Random Forest
    """

    predictions = model.predict(X_test)

    return predictions