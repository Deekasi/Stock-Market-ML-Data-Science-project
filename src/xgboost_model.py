"""
=========================================================
XGBoost Regression Model
=========================================================
"""

from xgboost import XGBRegressor


def train_xgboost(X_train, y_train):
    """
    Train an XGBoost regression model.
    """

    print("=" * 80)
    print("TRAINING XGBOOST")
    print("=" * 80)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print("\nXGBoost training completed successfully.")

    return model


def predict_xgboost(model, X_test):
    """
    Generate predictions using trained XGBoost model.
    """

    predictions = model.predict(X_test)

    return predictions