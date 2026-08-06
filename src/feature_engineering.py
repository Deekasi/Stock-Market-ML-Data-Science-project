import pandas as pd
import numpy as np


def add_features(df):

    df = df.copy()


    # Moving averages

    df["MA_20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )


    df["MA_50"] = (
        df["Close"]
        .rolling(window=50)
        .mean()
    )


    # Daily return

    df["Daily_Return"] = (
        df["Close"]
        .pct_change()
    )


    # RSI calculation

    delta = df["Close"].diff()

    gain = delta.where(
        delta > 0,
        0
    )

    loss = -delta.where(
        delta < 0,
        0
    )


    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )


    rs = avg_gain / avg_loss

    df["RSI"] = (
        100 -
        (100 / (1 + rs))
    )


    # MACD

    ema12 = (
        df["Close"]
        .ewm(span=12)
        .mean()
    )

    ema26 = (
        df["Close"]
        .ewm(span=26)
        .mean()
    )


    df["MACD"] = ema12 - ema26


    # Bollinger Bands

    rolling_mean = (
        df["Close"]
        .rolling(20)
        .mean()
    )


    rolling_std = (
        df["Close"]
        .rolling(20)
        .std()
    )


    df["Upper_Band"] = (
        rolling_mean + 
        (2 * rolling_std)
    )


    df["Lower_Band"] = (
        rolling_mean -
        (2 * rolling_std)
    )


    # Lag features

    df["Close_Lag_1"] = (
        df["Close"]
        .shift(1)
    )


    df["Close_Lag_5"] = (
        df["Close"]
        .shift(5)
    )


    # Remove empty rows

    df.dropna(
        inplace=True
    )


    return df