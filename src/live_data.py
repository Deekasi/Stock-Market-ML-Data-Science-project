import yfinance as yf
import pandas as pd


def download_live_stock_data(
    ticker="AAPL",
    period="2y",
    interval="1d"
):
    """
    Download latest historical market data
    from Yahoo Finance.
    """

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(
            f"No market data returned for {ticker}"
        )

    # Handle MultiIndex columns returned by
    # some yfinance versions.
    if isinstance(df.columns, pd.MultiIndex):

        df.columns = [
            column[0]
            for column in df.columns
        ]

    df = df.reset_index()

    # Normalize Date column

    if "Date" not in df.columns:

        if "Datetime" in df.columns:

            df.rename(
                columns={
                    "Datetime": "Date"
                },
                inplace=True
            )

    # Keep required columns

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[
        required_columns
    ].copy()

    # Convert numeric columns

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Remove invalid rows

    df = df.dropna(
        subset=required_columns
    )

    df = df.sort_values(
        "Date"
    )

    df = df.reset_index(
        drop=True
    )

    return df