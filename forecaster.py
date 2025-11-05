import prophet
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scraper import get_stocks_list

def forecast_stock(stock):
    df = pd.read_csv(f"processed_data/{stock}.csv")
    # Split df into train (first 50%) and test (next 50%) sets
    split_len = int(len(df) * 0.7)
    train_df = df.iloc[:split_len]
    test_df = df.iloc[split_len:]

    # Rename columns to ds and y
    train_df = train_df.rename(columns={"Date": "ds", "Close": "y"})
    test_df = test_df.rename(columns={"Date": "ds", "Close": "y"})

    train_df['ds'] = pd.to_datetime(train_df['ds'])
    test_df['ds'] = pd.to_datetime(test_df['ds'])

    # Drop all other columns
    train_df = train_df.drop(columns=["High", "Low", "Open", "Volume", "20_day_return", "20_day_volatility", "200_day_MA", "Market_Regime", "relvol"])
    test_df = test_df.drop(columns=["High", "Low", "Open", "Volume", "20_day_return", "20_day_volatility", "200_day_MA", "Market_Regime", "relvol"])

    model = prophet.Prophet()
    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df), freq="D")
    forecast = model.predict(future)
    forecast_test = forecast.iloc[-len(test_df):][['ds', 'yhat']]

    merged = test_df.merge(forecast_test, on='ds')

    # Compute accuracy metrics
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    rmse = mean_squared_error(merged['y'], merged['yhat'])
    mape = mean_absolute_percentage_error(merged['y'], merged['yhat'])

    print(f"Performance for {stock}:")
    print(f"  MAE  = {mae:.2f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAPE = {mape * 100:.2f}%")
    



if __name__ == "__main__":
    for stock in get_stocks_list():
        forecast_stock(stock)

    