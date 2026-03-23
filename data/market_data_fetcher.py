import os
import pandas as pd
import numpy as np
import yfinance as yf
import logging

# Configure logging to show time, level, and message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class MarketDataFetcher:
    """
    A professional data pipeline to fetch, clean, and process historical 
    market data for quantitative modeling.
    """
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        
    def fetch_and_process(self) -> dict:
        processed_data = {}
        
        for ticker in self.tickers:
            logging.info(f"Initiating data fetch for {ticker}...")
            
            try:
                # 1. Wrapped Download in Try-Except
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                
                if df.empty:
                    logging.warning(f"No data returned for {ticker} within the specified range.")
                    continue
                
                # 2. Minimum Data Requirement Check
                # We need at least 30 days of data for the rolling volatility window
                if len(df) < 30:
                    logging.error(f"Insufficient data for {ticker}. Found {len(df)} rows, need at least 30.")
                    continue

                # Handle multi-index columns from yfinance
                price_col = 'Adj Close' if 'Adj Close' in df.columns.get_level_values(0) else 'Close'
                if isinstance(df.columns, pd.MultiIndex):
                    df = df[price_col].copy()
                    df = df.iloc[:, 0].to_frame() if isinstance(df, pd.DataFrame) else df.to_frame()
                else:
                    df = df[[price_col]].copy()
                
                df.columns = ['Price']
                df.ffill(inplace=True)
                df.dropna(inplace=True)

                # 3. Processing Mathematical Metrics
                df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
                
                # Annualized Volatility calculation:
                # $$\sigma_{ann} = \text{std}(\text{returns}) \times \sqrt{252}$$
                df['Rolling_Volatility_30D'] = df['Log_Return'].rolling(window=30).std() * np.sqrt(252)
                
                # Clean up NaN from shifts and rolling windows
                df.dropna(inplace=True)
                
                # Final check after processing
                if df.empty:
                    logging.warning(f"Processing {ticker} resulted in an empty dataset (likely due to NaNs).")
                    continue

                processed_data[ticker] = df
                
                # Save results
                csv_path = os.path.join(self.data_dir, f"{ticker}_historical_data.csv")
                df.to_csv(csv_path)
                logging.info(f"Successfully saved {ticker} data to: {csv_path}")

            except Exception as e:
                # Catch-all for network issues, API changes, or disk permission errors
                logging.error(f"An unexpected error occurred while processing {ticker}: {str(e)}")
            
        return processed_data

if __name__ == "__main__":
    target_equities = ['ALV.DE', 'SIE.DE'] # Allianz and Siemens
    
    fetcher = MarketDataFetcher(
        tickers=target_equities, 
        start_date="2021-01-01", 
        end_date="2026-03-23" 
    )
    
    market_data = fetcher.fetch_and_process()
    logging.info("Pipeline execution complete.")