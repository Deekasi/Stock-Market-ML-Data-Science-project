"""
=========================================================
PREDICTION MODULE
=========================================================

Phase 11

Purpose:
    Load the saved model, scaler and feature columns
    and generate a next-day closing price prediction.
=========================================================
"""

import os
import joblib
import pandas as pd


# =====================================================
# FILE PATHS
# =====================================================

MODEL_PATH = "models/trained_models/linear_regression_model.pkl"

SCALER_PATH = "models/trained_models/scaler.pkl"

FEATURES_PATH = "models/trained_models/feature_columns.pkl"


# =====================================================
# LOAD MODEL
# =====================================================

def load_saved_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# =====================================================
# LOAD SCALER
# =====================================================

def load_saved_scaler():

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Saved scaler not found: {SCALER_PATH}"
        )

    scaler = joblib.load(SCALER_PATH)

    return scaler


# =====================================================
# LOAD FEATURE COLUMNS
# =====================================================

def load_feature_columns():

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Feature columns not found: {FEATURES_PATH}"
        )

    feature_columns = joblib.load(
        FEATURES_PATH
    )

    return feature_columns


# =====================================================
# PREPARE LATEST DATA
# =====================================================

def prepare_latest_data(
    df,
    feature_columns
):
    """
    Prepare the latest row for prediction.
    """

    data = df.copy()

    # Remove columns that aren't model features
    data = data.drop(
        columns=[
            "Date",
            "Target"
        ],
        errors="ignore"
    )

    # Check missing features
    missing_columns = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing feature columns: "
            + str(missing_columns)
        )

    # Keep exactly the same features
    # used during training
    data = data[feature_columns]

    # Select latest row
    latest_row = data.tail(1)

    # Replace infinite values
    latest_row = latest_row.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    # Check missing values
    if latest_row.isnull().any().any():

        missing = latest_row.columns[
            latest_row.isnull().any()
        ].tolist()

        raise ValueError(
            "Latest row contains missing values: "
            + str(missing)
        )

    return latest_row


# =====================================================
# MAKE PREDICTION
# =====================================================

def predict_next_close(
    df,
    model,
    scaler,
    feature_columns
):
    """
    Predict the next day's closing price.
    """

    latest_data = prepare_latest_data(
        df,
        feature_columns
    )

    # Apply SAME scaler used during training
    latest_scaled = scaler.transform(
        latest_data
    )

    # Predict
    prediction = model.predict(
        latest_scaled
    )

    return float(prediction[0])