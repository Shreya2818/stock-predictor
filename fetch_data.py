import yfinance as yf
import pandas as pd

# Use valid period = '1d' for 1-minute data
df = yf.download("RPOWER.NS", interval="1m", period="1d")

# Just to be safe: drop missing or empty rows
df.dropna(inplace=True)

# Preview the result
print(df.tail(5))
print("Data shape:", df.shape)
