"""
Module: data_loader.py

Purpose:
    Download stock market data from Yahoo Finance
    and save it as a CSV file.

Author: Deeksha
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str = None
) -> pd.DataFrame:
    """
    Download historical stock data.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g., AAPL, TSLA).
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date. If None, today's date is used.

    Returns
    -------
    pd.DataFrame
        Historical stock data.
    """

    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        progress=False
    )

    return df


def save_raw_data(
    df: pd.DataFrame,
    ticker: str,
    save_path: str
) -> None:
    """
    Save raw data to CSV.
    """

    Path(save_path).mkdir(parents=True, exist_ok=True)

    file_path = Path(save_path) / f"{ticker}.csv"

    df.to_csv(file_path)

    print(f"Raw data saved to: {file_path}")