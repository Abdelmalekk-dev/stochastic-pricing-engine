import yfinance as yf
import pandas as pd
import numpy as np
import os

def fetch_european_equities(tickers, start_date, end_date, output_dir="data"):
    """
    Fetches historical daily adjusted closing prices for given tickers,
    handles missing values, calculates log returns, and exports a clean CSV.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Fetching data for: {', '.join(tickers)}")
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Format handling for single vs multiple tickers
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
        
    # Data Cleaning: Forward fill for trading holidays, drop any remaining NaNs at the start
    print("Cleaning data and handling missing values...")
    clean_data = data.ffill().dropna()
    
    # Calculate daily logarithmic returns
    print("Calculating daily log returns...")
    returns = np.log(clean_data / clean_data.shift(1)).dropna()
    
    # Rename columns for clarity before merging
    clean_data.columns = [f"{col}_Price" for col in clean_data.columns]
    returns.columns = [f"{col}_LogReturn" for col in returns.columns]
    
    # Combine prices and returns into a single DataFrame
    final_df = pd.concat([clean_data, returns], axis=1).dropna()
    
    # Export to CSV in the data/ folder
    output_path = os.path.join(output_dir, "european_equities_cleaned.csv")
    final_df.to_csv(output_path)
    print(f"Data successfully saved to {output_path}\n")
    
    return final_df

if __name__ == "__main__":
    # Target equities: Allianz and Siemens
    target_tickers = ["ALV.DE", "SIE.DE"]
    
    # Fetch the last few years of data
    df = fetch_european_equities(
        tickers=target_tickers,
        start_date="2020-01-01",
        end_date="2025-12-31" 
    )
    
    # Display the first few rows to verify
    print(df.head())
