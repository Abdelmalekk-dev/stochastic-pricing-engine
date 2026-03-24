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
    market data for quantitative modeling, calculating parameters for both
    P (Real-world) and Q (Risk-neutral) probability measures.
    """
    def __init__(self, tickers: list, start_date: str, end_date: str, risk_free_rate: float = 0.02):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
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
                # We need at least 252 days of data for stable drift calculations
                if len(df) < 252:
                    logging.error(f"Insufficient data for {ticker}. Found {len(df)} rows, need at least 252 for annual drift.")
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
                
                # Volatility (\sigma): Rolling 30-day annualized
                df['Rolling_Volatility_30D'] = df['Log_Return'].rolling(window=30).std() * np.sqrt(252)
                
                # To calculate stable drift (\mu), we use a 252-day (1 year) rolling window.
                # \mu = Annualized Mean Log Return + 0.5 * \sigma^2 (Ito's Lemma adjustment)
                rolling_vol_252 = df['Log_Return'].rolling(window=252).std() * np.sqrt(252)
                rolling_mean_log_ret_252 = df['Log_Return'].rolling(window=252).mean() * 252
                
                # Historical Drift under measure P
                df['Realized_Drift_P'] = rolling_mean_log_ret_252 + 0.5 * (rolling_vol_252**2)
                
                # 4. Girsanov's Theorem: The Market Price of Risk (\theta)
                # \theta = (\mu - r) / \sigma
                df['Market_Price_of_Risk'] = (df['Realized_Drift_P'] - self.risk_free_rate) / rolling_vol_252
                
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
                logging.error(f"An unexpected error occurred while processing {ticker}: {str(e)}")
            
        return processed_data

if __name__ == "__main__":
    target_equities = ['ALV.DE', 'SIE.DE'] 
    
    # Assuming a 2% risk-free rate (r = 0.02)
    fetcher = MarketDataFetcher(
        tickers=target_equities, 
        start_date="2021-01-01", 
        end_date="2026-03-23",
        risk_free_rate=0.02 
    )
    
    market_data = fetcher.fetch_and_process()
    logging.info("Pipeline execution complete.")