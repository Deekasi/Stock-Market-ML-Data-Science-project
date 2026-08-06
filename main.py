"""
Main entry point for Stock Market Prediction Project.

Pipeline:
1. Fetch stock data using yfinance
2. Perform EDA
3. Generate visualizations
4. Feature Engineering
5. Save processed dataset
"""


# ===============================
# Import Data Loader
# ===============================

from src.data_loader import load_stock_data


# ===============================
# Import EDA Functions
# ===============================

from src.eda import (
    basic_information,
    check_missing_values,
    check_duplicates,
    summary_statistics,
)


# ===============================
# Import Visualization Functions
# ===============================

from src.visualization import (
    plot_close_price,
    plot_volume,
    plot_histogram,
    plot_boxplot,
    plot_correlation_heatmap,
)


# ===============================
# Import Feature Engineering
# ===============================

from src.feature_engineering import add_features




def main():

    """
    Complete stock prediction pipeline.
    """


    # ===============================
    # Step 1: Download Stock Data
    # ===============================

    ticker = "AAPL"

    print("\n===== Downloading Stock Data =====")


    df = load_stock_data(ticker)



    print("\nRaw Data Preview:")

    print(df.head())



    # ===============================
    # Step 2: Exploratory Data Analysis
    # ===============================

    print("\n===== Performing EDA =====")


    basic_information(df)


    check_missing_values(df)


    check_duplicates(df)


    summary_statistics(df)



    # ===============================
    # Step 3: Data Visualization
    # ===============================

    print("\n===== Creating Visualizations =====")


    plot_close_price(df)


    plot_volume(df)


    plot_histogram(df)


    plot_boxplot(df)


    plot_correlation_heatmap(df)



    # ===============================
    # Step 4: Feature Engineering
    # ===============================

    print("\n===== Feature Engineering =====")


    df = add_features(df)



    print("\nFeature Engineered Columns:")

    print(df.columns)



    print("\nUpdated Dataset:")

    print(df.head())



    # ===============================
    # Step 5: Save Processed Data
    # ===============================

    output_file = (
        f"data/processed/{ticker}_features.csv"
    )


    df.to_csv(
        output_file,
        index=False
    )


    print("\nProcessed data saved:")
    print(output_file)




if __name__ == "__main__":

    main()