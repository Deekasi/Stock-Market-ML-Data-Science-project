"""
Feature Scaling Module
"""

from sklearn.preprocessing import StandardScaler


def scale_features(X_train, X_test):
    """
    Scale training and testing features.
    """

    scaler = StandardScaler()

    # Learn from training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same transformation to test data
    X_test_scaled = scaler.transform(X_test)

    print("=" * 80)
    print("FEATURE SCALING")
    print("=" * 80)

    print("\nTraining Data Shape :", X_train_scaled.shape)
    print("Testing Data Shape  :", X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, scaler