import os
import yfinance as yf
import pandas as pd

def fetch_market_data():
    """
    Fetches US10Y (^TNX) and VIX (^VIX) data from Yahoo Finance
    on a 1D timeframe, for the date range: 30-04-2026 to 30-05-2026.
    Saves the data as CSV files in the 'may' folder.
    """
    start_date = "2026-04-30"
    end_date = "2026-05-31"  # to include 30-05-2026 date, end date must be the day after
    
    # Save to the existing 'May_data' folder
    output_dir = "May_data"
    
    # ^TNX: CBOE Interest Rate 10 Year T No
    # ^VIX: CBOE Volatility Index
    tickers = {
        "US10Y": "^TNX",
        "VIX": "^VIX"
    }
    
    print(f"Fetching 1D data from {start_date} to {end_date}...")
    
    for name, ticker in tickers.items():
        print(f"Downloading {name} ({ticker})...")
        # Fetch data for each ticker
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        
        if data.empty:
            print(f"No data was fetched for {name}.")
            continue
            
        # Define output path
        file_path = os.path.join(output_dir, f"{name.lower()}_1d.csv")
        
        # Save to CSV
        data.to_csv(file_path)
        print(f"Saved {name} data to {file_path}")
        
    print("\nData fetched and saved successfully!")

if __name__ == "__main__":
    fetch_market_data()
