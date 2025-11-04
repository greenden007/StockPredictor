import yfinance as yf
import sys

def download_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)