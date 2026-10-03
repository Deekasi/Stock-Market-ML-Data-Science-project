"""
=========================================================
STOCK MARKET PREDICTION PROJECT
=========================================================

PHASE 11
REAL-WORLD PREDICTION & MODEL TESTING

Target:
    Tomorrow's Close price

Final Model:
    Linear Regression

=========================================================
"""

# =====================================================
# IMPORTS
# =====================================================

import os
import joblib

from src.eda import (
    load_data,
    basic_information,
    check_missing_values,
    check_duplicates,
    summary_statistics,
)

from src.feature_engineering import (
    create_moving_averages,
    create_daily_returns,
    create_lag_features,
    create_rsi,
    create_macd,
    create_bollinger_bands,
    create_date_features,
)

from src.preprocessing import (
    clean_dataset,
)

from src.model_preparation import (
    prepare_features_and_target,
)

from src.predict import (
    load_saved_model,
    load_saved_scaler,
    load_feature_columns,
    predict_next_close,
)


# =====================================================
# MAIN FUNCTION
# =====================================================

def main():

    # =================================================
    # CONFIGURATION
    # =================================================

    FILE_PATH = "data/raw/AAPL.csv"

    MODEL_DIRECTORY = (
        "models/trained_models"
    )

    MODEL_PATH = (
        f"{MODEL_DIRECTORY}/"
        "linear_regression_model.pkl"
    )

    SCALER_PATH = (
        f"{MODEL_DIRECTORY}/"
        "scaler.pkl"
    )

    FEATURES_PATH = (
        f"{MODEL_DIRECTORY}/"
        "feature_columns.pkl"
    )


    # =================================================
    # HEADER
    # =================================================

    print("\n")

    print("=" * 80)
    print("STOCK MARKET PREDICTION PROJECT")
    print("=" * 80)

    print("\nPHASE 11")
    print("REAL-WORLD PREDICTION & MODEL TESTING")


    # =================================================
    # STEP 1 — CHECK SAVED FILES
    # =================================================

    print("\n")
    print("=" * 80)
    print("STEP 1 — CHECK SAVED MODEL FILES")
    print("=" * 80)

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH
    ]

    for file_path in required_files:

        if os.path.exists(file_path):

            print(
                f"✓ Found: {file_path}"
            )

        else:

            raise FileNotFoundError(
                f"\nRequired file missing: {file_path}"
                "\nRun Phase 10D first."
            )


    # =================================================
    # STEP 2 — LOAD SAVED MODEL
    # =================================================

    print("\n")
    print("=" * 80)
    print("STEP 2 — LOAD SAVED MODEL")
    print("=" * 80)

    model = load_saved_model()

    scaler = load_saved_scaler()

    feature_columns = (
        load_feature_columns()
    )

    print(
        "\n✓ Linear Regression model loaded."
    )

    print(
        "✓ Scaler loaded."
    )

    print(
        "✓ Feature columns loaded."
    )


    # =================================================
    # STEP 3 — LOAD RAW DATA
    # =================================================

    print("\n")
    print("=" * 80)
    print("STEP 3 — LOAD LATEST STOCK DATA")
    print("=" * 80)

    df = load_data(
        FILE_PATH
    )

    print(
        "\nRaw dataset shape:",
        df.shape
    )


    # =================================================
    # STEP 4 — CLEAN DATA
    # =================================================

    print("\n")
    print("=" * 80)
    print("STEP 4 — DATA CLEANING")
    print("=" * 80)

    df = clean_dataset(df)

    print(
        "\nCleaned dataset shape:",
        df.shape
    )


    # =================================================
    # STEP 5 — FEATURE ENGINEERING
    # =================================================

    print("\n")
    print("=" * 80)
    print("STEP 5 — FEATURE ENGINEERING")
    print("=" * 80)

    df = create_moving_averages(df)

    df = create_daily_returns(df)

    df = create_lag_features(df)

    df = create_rsi(df)

    df = create_macd(df)

    df = create_bollinger_bands(df)

    df = create_date_features(df)

    print(
        "\n✓ Feature engineering completed."
    )

    print(
        "Engineered dataset shape:",
        df.shape
    )


    # =================================================
    # STEP 6 — CLEAN ENGINEERED DATA
    # =================================================

    print("\n")
    print("=" * 80)
    print("STEP 6 — FINAL DATA PREPARATION")
    print("=" * 80)

    df = clean_dataset(df)

    print(
        "\n✓ Final prediction dataset prepared."
    )


    # =================================================
    # STEP 7 — SHOW LATEST DATA
    # =================================================

    print("\n")
    print("=" * 80)
    print("STEP 7 — LATEST MARKET DATA")
    print("=" * 80)

    print(
        "\nLast 5 rows:"
    )

    print(
        df.tail()
    )


    # =================================================
    # STEP 8 — CURRENT CLOSE
    # =================================================

    print("\n")
    print("=" * 80)
    print("STEP 8 — CURRENT CLOSE PRICE")
    print("=" * 80)

    current_close = (
        df["Close"].iloc[-1]
    )

    print(
        "\nLatest available Close:",
        round(
            float(current_close),
            2
        )
    )


    # =================================================
    # STEP 9 — NEXT-DAY PREDICTION
    # =================================================

    print("\n")
    print("=" * 80)
    print("STEP 9 — NEXT-DAY PREDICTION")
    print("=" * 80)

    predicted_close = predict_next_close(
        df,
        model,
        scaler,
        feature_columns
    )

    print(
        "\nPredicted next-day Close:"
    )

    print(
        round(
            predicted_close,
            2
        )
    )


    # =================================================
    # STEP 10 — PREDICTION DIFFERENCE
    # =================================================

    print("\n")
    print("=" * 80)
    print("STEP 10 — PREDICTION ANALYSIS")
    print("=" * 80)

    difference = (
        predicted_close
        - float(current_close)
    )

    percentage_change = (
        difference
        / float(current_close)
    ) * 100


    print(
        "\nCurrent Close:",
        round(
            float(current_close),
            2
        )
    )

    print(
        "Predicted Close:",
        round(
            predicted_close,
            2
        )
    )

    print(
        "Predicted Change:",
        round(
            difference,
            2
        )
    )

    print(
        "Predicted Change %:",
        round(
            percentage_change,
            2
        ),
        "%"
    )


    # =================================================
    # STEP 11 — INTERPRETATION
    # =================================================

    print("\n")
    print("=" * 80)
    print("STEP 11 — MODEL INTERPRETATION")
    print("=" * 80)

    if difference > 0:

        print(
            "\nModel prediction indicates "
            "an upward movement."
        )

    elif difference < 0:

        print(
            "\nModel prediction indicates "
            "a downward movement."
        )

    else:

        print(
            "\nModel prediction indicates "
            "almost no change."
        )


    # =================================================
    # FINAL RESULT
    # =================================================

    print("\n")
    print("=" * 80)
    print("PHASE 11 COMPLETED")
    print("=" * 80)

    print(
        "\nFinal Model:"
    )

    print(
        "Linear Regression"
    )

    print(
        "\nCurrent Close:",
        round(
            float(current_close),
            2
        )
    )

    print(
        "Predicted Next-Day Close:",
        round(
            predicted_close,
            2
        )
    )

    print(
        "\nSaved model successfully used "
        "for independent prediction."
    )

    print("\n")


# =====================================================
# PROGRAM ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()