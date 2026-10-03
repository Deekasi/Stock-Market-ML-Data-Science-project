from src.live_data import download_live_stock_data


df = download_live_stock_data(
    ticker="AAPL",
    period="2y"
)

print(df.tail())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)