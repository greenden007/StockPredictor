import yfinance as yf
from datetime import datetime, timedelta

def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(f"data/{ticker}.csv")

def get_stocks_list():
    with open("stocks.txt", "r") as f:
        stocks = f.read().splitlines()
    return stocks

def main():
    stocks = get_stocks_list()
    years = 8
    for stock in stocks:
        get_data(stock, datetime.now() - timedelta(days=365*years), datetime.now())

if __name__ == "__main__":
    main()
