import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_close_price(df):
    """
    Plot the Close Price over time.
    """

    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(15, 6))

    plt.plot(
        df["Date"],
        df["Close"],
        color="blue",
        linewidth=2,
        label="Close Price"
    )

    plt.title("Apple Stock Closing Price")

    plt.xlabel("Date")

    plt.ylabel("Price (USD)")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig("reports/figures/close_price.png")

    plt.show()
def plot_volume(df):
    """
    Plot daily trading volume.
    """

    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(15, 5))

    plt.plot(
        df["Date"],
        df["Volume"],
        color="green",
        linewidth=1
    )

    plt.title("Daily Trading Volume")

    plt.xlabel("Date")

    plt.ylabel("Volume")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("reports/figures/volume.png")

    plt.show()
def plot_histogram(df):
    """
    Plot histogram of closing prices.
    """

    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(10, 5))

    plt.hist(
        df["Close"],
        bins=30,
        edgecolor="black"
    )

    plt.title("Distribution of Closing Prices")

    plt.xlabel("Closing Price")

    plt.ylabel("Frequency")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("reports/figures/histogram.png")

    plt.show()        
def plot_boxplot(df):
    """
    Plot boxplot for closing prices.
    """

    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(6, 5))

    plt.boxplot(df["Close"])

    plt.title("Box Plot of Closing Prices")

    plt.ylabel("Price")

    plt.tight_layout()

    plt.savefig("reports/figures/boxplot.png")

    plt.show()
def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap.
    """

    os.makedirs("reports/figures", exist_ok=True)

    correlation = df[["Open", "High", "Low", "Close", "Volume"]].corr()

    plt.figure(figsize=(8, 6))

    plt.imshow(correlation, cmap="coolwarm")

    plt.colorbar()

    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)

    plt.yticks(range(len(correlation.columns)), correlation.columns)

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig("reports/figures/correlation_heatmap.png")

    plt.close()
