import os
import pandas as pd
import numpy as np
import yfinance as yf

class MarketDataFetcher:
    """
    A robust data pipeline to fetch, clean, and process historical 
    market data for quantitative modeling.
    """
    def __init__(self, tickers: list, start_date: str, end_date: str):
        """
        Initializes the fetcher with target equities and timeframes.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
        # Dynamically locate the data/ directory relative to this script
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        
    def fetch_and_process(self) -> dict:
        """
        Downloads data, calculates logarithmic returns, and computes annualized volatility.
        Saves the clean data as CSV files.
        """
        processed_data = {}
        
        for ticker in self.tickers:
            print(f"Fetching data for {ticker}...")
            
            # Download historical daily data
            df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            
            if df.empty:
                print(f"Warning: No data found for {ticker}.")
                continue
            
            # Standardize column selection (handling yfinance's backend changes)
            # Check for 'Adj Close' first, fallback to 'Close' if missing
            price_col = 'Adj Close' if 'Adj Close' in df.columns.get_level_values(0) else 'Close'
            
            if isinstance(df.columns, pd.MultiIndex):
                # Extract the target price column
                df = df[price_col].copy()
                # If yfinance left the ticker name as a second column level, isolate the series
                if isinstance(df, pd.DataFrame):
                    df = df.iloc[:, 0].to_frame()
                else:
                    df = df.to_frame()
            else:
                df = df[[price_col]].copy()
                
            df.columns = ['Price']
            
            # Clean missing values: Forward fill first, then drop any remaining NaNs
            df.ffill(inplace=True)
            df.dropna(inplace=True)
            
            # Calculate daily logarithmic returns
            df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
            
            # Calculate rolling 30-day historical volatility (annualized)
            # Volatility = StdDev(log returns) * sqrt(252 trading days)
            df['Rolling_Volatility_30D'] = df['Log_Return'].rolling(window=30).std() * np.sqrt(252)
            
            # Drop the NaN values created by the rolling window and shift
            df.dropna(inplace=True)
            
            processed_data[ticker] = df
            
            # Export to CSV inside the data/ folder
            csv_path = os.path.join(self.data_dir, f"{ticker}_historical_data.csv")
            df.to_csv(csv_path)
            print(f"Saved cleaned data to: {csv_path}\n")
            
        return processed_data

if __name__ == "__main__":
    # Target German equities (Allianz and Siemens)
    target_equities = ['ALV.DE', 'SIE.DE']
    
    # Instantiate the pipeline for the last 5 years
    fetcher = MarketDataFetcher(
        tickers=target_equities, 
        start_date="2021-01-01", 
        end_date="2026-03-23" 
    )
    
    # Execute the pipeline
    market_data = fetcher.fetch_and_process()
    print("Market data ingestion and processing complete.")