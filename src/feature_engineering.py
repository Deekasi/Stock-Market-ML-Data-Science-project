"""
Feature Engineering Module
"""

import pandas as pd


def create_moving_averages(df):
    """
    Create Moving Average Features.
    """

    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()

    return df


def create_daily_returns(df):
    """
    Create Daily Return Feature.
    """

    df["Daily_Return"] = df["Close"].pct_change()

    return df


def create_lag_features(df):
    """
    Create Lag Features.
    """

    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)
    df["Lag_5"] = df["Close"].shift(5)
    df["Lag_10"] = df["Close"].shift(10)

    return df


def create_rsi(df, window=14):
    """
    Create RSI Feature.
    """

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()

    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def create_macd(df):
    """
    Create MACD Features.
    """

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()

    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    return df


def create_bollinger_bands(df):
    """
    Create Bollinger Bands.
    """

    df["BB_Middle"] = df["Close"].rolling(window=20).mean()

    std = df["Close"].rolling(window=20).std()

    df["BB_Upper"] = df["BB_Middle"] + (2 * std)

    df["BB_Lower"] = df["BB_Middle"] - (2 * std)

    return df


def create_date_features(df):
    """
    Create Date Features.
    """

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year

    df["Month"] = df["Date"].dt.month

    df["Day"] = df["Date"].dt.day

    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df["Quarter"] = df["Date"].dt.quarter

    df["Is_Month_Start"] = df["Date"].dt.is_month_start.astype(int)

    df["Is_Month_End"] = df["Date"].dt.is_month_end.astype(int)

    return df