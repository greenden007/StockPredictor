import numpy as np
import pandas as pd
from scraper import get_stocks_list

def get_stock_data(ticker):
    df = pd.read_csv(f"data/{ticker}.csv")
    # drop first two rows in place
    df = df.iloc[2:].reset_index(drop=True)
    return df

def add_interesting_stats(df):
    df['Date'] = pd.to_datetime(df['Date'])
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['20_day_return'] = df['Close'].pct_change(periods=20)
    df['20_day_volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    # Market Regime
    df['200_day_MA'] = df['Close'].rolling(window=200).mean()
    df['Market_Regime'] = np.where(
        df['Close'] > df['200_day_MA'] * 1.2, 'Bull',
        np.where(
            df['Close'] < df['200_day_MA'] * 0.8, 'Bear',
            'Neutral'
        )
    )

    df['relvol'] = df['20_day_volatility'] / df['20_day_volatility'].rolling(window=252, min_periods=1).mean()
    
    return df

if __name__ == "__main__":
    for stock in get_stocks_list():
        df = get_stock_data(stock)
        # rename price column to Date
        df = df.rename(columns={'Price': 'Date'})
        df = add_interesting_stats(df)
        df.to_csv(f"processed_data/{stock}.csv", index=False)
        
