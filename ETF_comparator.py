import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

def get_random_date_range(years_window=5):
    # Get a random start date between 2010 and (current_year - years_window)
    end_limit = datetime.now().year - years_window
    random_year = random.randint(2010, end_limit)
    random_month = random.randint(1, 12)
    random_day = random.randint(1, 28)  # Using 28 to avoid month-end issues
    
    start_date = datetime(random_year, random_month, random_day)
    end_date = start_date + timedelta(days=years_window*365)
    
    return start_date, end_date

def normalize_prices(df):
    # Normalize prices to start at 1
    return df / df.iloc[0]

# List of tickers to compare
tickers = ["VWRL.SW", "VT"]  # Add more tickers as needed

# Get random date range
start_date, end_date = get_random_date_range(5)

# Create empty DataFrame to store all prices
all_prices = pd.DataFrame()

# Download and process data for each ticker
for ticker in tickers:
    # Download historical data
    etf = yf.Ticker(ticker)
    data = etf.history(start=start_date, end=end_date)
    
    # Store only the Close prices
    all_prices[ticker] = data['Close']

# Drop any rows with NaN values (in case some tickers don't have data for certain dates)
all_prices = all_prices.dropna()

# Normalize all prices
normalized_prices = normalize_prices(all_prices)

# Create the plot
plt.figure(figsize=(12, 6))
for column in normalized_prices.columns:
    plt.plot(normalized_prices.index, normalized_prices[column], label=column)

plt.title(f'Normalized Price Comparison\n{start_date.date()} to {end_date.date()}')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print some performance metrics
print("\nPerformance Metrics:")
print("-" * 50)
for column in normalized_prices.columns:
    total_return = (normalized_prices[column].iloc[-1] - 1) * 100
    print(f"{column}: {total_return:.2f}% total return")
