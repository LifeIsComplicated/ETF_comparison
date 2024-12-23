import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random


def get_ticker_history(ticker, start_date=datetime(2000,1,1), end_date=datetime.now()):
    # Download historical data
    etf = yf.Ticker(ticker)
    data = etf.history(start=start_date, end=end_date)
    # Strip the hour to avoid UTC conversion issues and things like that
    data.index = data.index.date
    return data

def get_forex_history(currency_from, currency_to, start_date, end_date):
    # Add a small buffer to ensure we get all needed dates
    start_date = start_date - pd.Timedelta(days=1)
    end_date = end_date + pd.Timedelta(days=1)
    if currency_from == currency_to:
        return pd.Series(1, index=pd.date_range(start=start_date, end=end_date))
    
    forex = yf.Ticker(f"{currency_from}{currency_to}=X")
    forex_data = forex.history(start=start_date, end=end_date)
    
    # Strip the hour to avoid UTC conversion issues and things like that
    forex_data.index = forex_data.index.date
    return forex_data['Close']

def convert_all_prices_to_target_currency(df, tickers, operation_currencies, dividends_currencies, target_currency):
    # Ensure DataFrame index is datetime
    df.index = pd.to_datetime(df.index)

    for ticker in tickers:
        # Get the operation currency and dividend currency for the current ticker
        operation_currency = operation_currencies[ticker]
        dividend_currency = dividends_currencies[ticker]

        # Get the conversion rates for the current ticker
        conversion_dividends_to_target = get_forex_history(dividend_currency, target_currency, 
                                                      df.index.min(), df.index.max())
        conversion_operation_to_target = get_forex_history(operation_currency, target_currency, 
                                                      df.index.min(), df.index.max())
        
        # Align the forex data with the main dataframe
        conversion_dividends_to_target = conversion_dividends_to_target.reindex(df.index)
        conversion_operation_to_target = conversion_operation_to_target.reindex(df.index)
        

        # store conversion factors (for comparison)
        df[f"conversion_dividends_to_target_{ticker}"] = conversion_dividends_to_target
        df[f"conversion_operation_to_target_{ticker}"] = conversion_operation_to_target

        # store original prices (for comparison)
        df[f"Dividends_orig_{ticker}"] = df[f"Dividends_{ticker}"].copy()
        df[f"Close_orig_{ticker}"] = df[f"Close_{ticker}"].copy()

        # Convert the prices for the current ticker
        df[f"Dividends_{ticker}"] = df[f"Dividends_{ticker}"] * conversion_dividends_to_target
        df[f"Close_{ticker}"] = df[f"Close_{ticker}"] * conversion_operation_to_target
    
    return df

def cumsum_dividends(df):
    # Sum dividends for each date
    for column in df.columns:
        if "Dividends" in column:
            df[column] = df[column].cumsum()
    return df

def update_prices_with_dividends(df):
    for column in df.columns:
        if "Close" in column:
            ticker = column.split('_')[1]
            dividends_column = f'Dividends_{ticker}'
            if dividends_column in df.columns:
                df[column] = df[column] + df[dividends_column]
    return df

def normalize_prices(df):
    # Normalize prices (not dividends!) to start at 1 (for a fair comparison)
    for column in df.columns:
        if "Close" in column:
            df[column] = df[column] / df[column].iloc[0]
    return df

### backtester functions

def get_random_date_range(all_prices, years_window=5):
    try:
        # Safely get the first and last dates from the index
        first_date = all_prices.index.min()
        last_date = all_prices.index.max()
        
        # Get the year limits
        init_limit = first_date.year
        end_limit = last_date.year - years_window
        
        # Ensure end_limit is not less than init_limit
        if end_limit < init_limit:
            raise ValueError("Date range too small for the specified window")
        
        random_year = random.randint(init_limit, end_limit)
        random_month = random.randint(1, 12)
        random_day = random.randint(1, 28)  # Using 28 to avoid month-end issues

        start_date = datetime(random_year, random_month, random_day)
        end_date = start_date + timedelta(days=years_window*365)
        return start_date, end_date
        
    except (AttributeError, KeyError) as e:
        raise ValueError("Invalid or empty price data provided") from e


def get_performance(df, ticker):
    return (df.iloc[-1][f"Close_{ticker}"] - df.iloc[0][f"Close_{ticker}"])/df.iloc[0][f"Close_{ticker}"]

def backtester(all_prices, tickers):
    # Get a random date range
    start_date, end_date = get_random_date_range(all_prices, years_window=5)
    all_prices_bt = all_prices.loc[start_date:end_date]
    all_prices_bt = cumsum_dividends(all_prices_bt)
    all_prices_bt = update_prices_with_dividends(all_prices_bt)
    all_prices_bt = normalize_prices(all_prices_bt)
    if len(tickers)==2:
        return get_performance(all_prices_bt, tickers[1]) - get_performance(all_prices_bt, tickers[0])
    elif len(tickers)==1:
        return get_performance(all_prices_bt, tickers[0])
    else:
        print("More than 2 tickers listed. I don't know what comparison you'd like me to do!")

def plot_performance_histogram(performances):
    plt.figure(figsize=(10, 6))
    plt.hist(performances, bins=20, edgecolor='black')
    plt.title('Distribution of Performance Differences')
    plt.xlabel('Performance Difference')
    plt.ylabel('Frequency')
    
    # Add mean and median lines
    mean_perf = np.mean(performances)
    median_perf = np.median(performances)
    plt.axvline(mean_perf, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_perf:.3f}')
    plt.axvline(median_perf, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_perf:.3f}')
    
    # Add some statistics in a text box
    stats_text = f'Mean: {mean_perf:.3f}\nMedian: {median_perf:.3f}\nStd: {np.std(performances):.3f}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_normalized_prices(all_prices, tickers, start_date, end_date):
    # Create the plot
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.plot(all_prices.index, all_prices[f'Close_{ticker}'], label=f'{ticker} Close price')

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



####### GLOBAL VARIABLES ########
#################################
# List of tickers to compare
tickers = ["VWRL.SW", "VT"]  # Add more tickers as needed
operation_currencies = {}
operation_currencies["VWRL.SW"] = "CHF"
operation_currencies["VT"] = "USD"

dividends_currencies = {}
dividends_currencies["VWRL.SW"] = "USD"
dividends_currencies["VT"] = "USD"

target_currency = "CHF"

n_backtests = 1000
random_int_for_price_comparison = random.randint(0, n_backtests)


if __name__ == "__main__":
    ## initialize the dataframe
    for ticker in tickers:
        all_prices = pd.DataFrame(columns=[f'Close_{ticker}', f'Dividends_{ticker}'])

    # Download and process data for each ticker and populate the dataframe
    for ticker in tickers:
        # Download historical data
        data = get_ticker_history(ticker=ticker)
        all_prices[f'Close_{ticker}'] = data['Close'] 
        all_prices[f'Dividends_{ticker}'] = data['Dividends'] 
        
    all_prices = convert_all_prices_to_target_currency(all_prices, tickers, operation_currencies, dividends_currencies, target_currency)
    all_prices = all_prices.dropna()

    if len(all_prices) == 0:
        print("No valid data found for the specified tickers and date range.")
        exit(1)

    #### Start backtests
    performances = list()
    for i in range(n_backtests):
        performances.append(backtester(all_prices, tickers))
        if i == random_int_for_price_comparison:
            plot_normalized_prices(all_prices, tickers, all_prices.index.min(), all_prices.index.max())

    plot_performance_histogram(performances)
 