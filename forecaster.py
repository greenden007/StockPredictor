import prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scraper import get_stocks_list
import os
import multiprocessing as mp
from functools import partial
import warnings

# Suppress some common warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='prophet')

# Set the number of processes to use (leave one core free for system)
NUM_PROCESSES = max(1, mp.cpu_count() - 1)

def forecast_stock(stock, yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False):
    """
    Forecast stock prices using Prophet with rolling window.
    
    Args:
        stock: Stock symbol
        yearly_seasonality: Whether to include yearly seasonality
        weekly_seasonality: Whether to include weekly seasonality
        daily_seasonality: Whether to include daily seasonality
    """
    try:
        df = pd.read_csv(f"processed_data/{stock}.csv")
        
        # Convert Date to datetime and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Calculate split point (70% train, 30% test)
        split_idx = int(len(df) * 0.7)
        
        # Prepare the initial training set
        train_df = df[['Date', 'Close']].copy()
        train_df.columns = ['ds', 'y']
        
        # Initialize lists to store results
        predictions = []
        actuals = []
        
        # Initialize the model with specified seasonality
        model = prophet.Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=0.1  # More flexible trend
        )
        
        # Train on initial data (first 70%)
        initial_train = train_df.iloc[:split_idx]
        model.fit(initial_train)
        
        # Use the model to predict the entire test set at once for efficiency
        future_dates = train_df.iloc[split_idx:][['ds']].copy()
        forecast = model.predict(future_dates)
        
        # Get predictions and actuals
        predictions = forecast['yhat'].values
        actuals = train_df.iloc[split_idx:]['y'].values
        
        # Prepare forecast DataFrame in the expected format
        forecast_df = pd.DataFrame({
            'ds': future_dates['ds'].values,
            'yhat': predictions
        })
        
        # Calculate metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = mean_absolute_percentage_error(actuals, predictions)
        
        print(f"Performance for {stock}:")
        print(f"  MAE  = {mae:.2f}")
        print(f"  RMSE = {rmse:.2f}")
        print(f"  MAPE = {mape * 100:.2f}%")
        
        return mae, rmse, mape, forecast_df
        
    except Exception as e:
        print(f"Error in forecast_stock for {stock}: {str(e)}")
        raise
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = mean_absolute_percentage_error(actuals, predictions)
    
    print(f"Performance for {stock}:")
    print(f"  MAE  = {mae:.2f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAPE = {mape * 100:.2f}%")
    
    # Prepare forecast DataFrame in the expected format
    forecast_df = pd.DataFrame({
        'ds': train_df.iloc[split_idx:]['ds'],
        'yhat': predictions
    })
    
    return mae, rmse, mape, forecast_df

def game(forecast):
    """
    Make a trading decision based on the forecasted stock prices.
    
    Args:
        forecast: DataFrame containing forecasted values including 'yhat', 'yhat_lower', 'yhat_upper'
        
    Returns:
        int: -1 for sell, 1 for buy (no hold option)
    """
    # Get the last few forecasted values to analyze the trend
    last_forecasts = forecast['yhat'][-5:].values
    
    # Calculate the short-term trend (difference between last and first of last 5 predictions)
    trend = last_forecasts[-1] - last_forecasts[0]
    
    # Calculate the percentage change of the trend
    pct_change = trend / last_forecasts[0]
    
    # Make decision based on percentage change
    if pct_change > 0:  # If price is expected to increase
        return 1  # Buy signal
    else:  # If price is expected to stay the same or decrease
        return -1  # Sell signal

def get_game_accuracy(actual_prices, forecast):
    """
    Calculate the accuracy of the trading signals based on actual price movements.
    
    Args:
        actual_prices: Series of actual stock prices (should be the test portion)
        forecast: DataFrame containing forecasted values including 'yhat' for the test period
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    # Ensure we have matching lengths
    min_len = min(len(actual_prices), len(forecast))
    actual_prices = actual_prices.iloc[:min_len]
    forecast = forecast.iloc[:min_len]
    
    # Generate signals based on forecasted vs actual prices
    signals = []
    
    # For each day, decide whether to buy or sell based on the next day's forecast
    for i in range(len(forecast) - 1):
        current_price = actual_prices.iloc[i]
        next_forecast = forecast.iloc[i+1]['yhat'] if i+1 < len(forecast) else forecast.iloc[-1]['yhat']
        
        if next_forecast > current_price:
            signals.append(1)  # Buy signal
        else:
            signals.append(-1)  # Sell signal
    
    # Calculate actual price changes for the next day
    actual_changes = actual_prices.diff().shift(-1).dropna()
    
    # Calculate trading performance
    correct_predictions = 0
    total_predictions = len(signals)
    
    for i in range(min(len(signals), len(actual_changes))):
        signal = signals[i]
        actual_change = actual_changes.iloc[i]
        
        # Check if prediction was correct
        if (signal == 1 and actual_change > 0) or (signal == -1 and actual_change <= 0):
            correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Simulate trading
    initial_balance = 1000
    balance = initial_balance
    position = 0  # 0 = no position, 1 = long position
    
    for i in range(len(signals)):
        if i >= len(actual_prices) - 1:
            break
            
        current_price = actual_prices.iloc[i]
        
        if signals[i] == 1 and position == 0:  # Buy signal and no position
            position = balance / current_price  # Buy as many shares as possible
            balance = 0
        elif signals[i] == -1 and position > 0:  # Sell signal and have position
            balance = position * current_price
            position = 0
    
    # Close any open position at the last price
    if position > 0:
        balance = position * actual_prices.iloc[-1]
    
    profit = balance - initial_balance
    profit_pct = (balance / initial_balance - 1) * 100
    
    return {
        'accuracy': accuracy,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'initial_balance': initial_balance,
        'final_balance': balance,
        'profit': profit,
        'profit_pct': profit_pct,
        'buy_signals': signals.count(1),
        'sell_signals': signals.count(-1)
    }

def process_stock(stock, results_queue=None):
    """Process a single stock and return its results."""
    try:
        print(f"Processing {stock}...")
        mae, rmse, mape, forecast = forecast_stock(stock)
        
        # Load actual prices for accuracy calculation
        df = pd.read_csv(f"processed_data/{stock}.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Get the test portion of the data (last 30%)
        split_idx = int(len(df) * 0.7)
        test_prices = df['Close'].iloc[split_idx:].reset_index(drop=True)
        
        # Calculate trading accuracy and performance
        accuracy_metrics = get_game_accuracy(test_prices, forecast)
        
        # Get the final recommendation based on the last prediction
        last_forecast = forecast.iloc[-1]['yhat']
        last_price = test_prices.iloc[-1] if len(test_prices) > 0 else df['Close'].iloc[-1]
        decision = 1 if last_forecast > last_price else -1
        action = 'BUY' if decision == 1 else 'SELL'
        
        print(f"\nRecommendation for {stock}: {action}")
        print(f"Trading Accuracy: {accuracy_metrics['accuracy']*100:.2f}%")
        print(f"Profit/Loss: ${accuracy_metrics['profit']:.2f} ({accuracy_metrics['profit_pct']:.2f}%)")
        print(f"Signals - Buy: {accuracy_metrics['buy_signals']}, Sell: {accuracy_metrics['sell_signals']}")
        
        result = {
            "Stock": stock,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Action": action,
            "Trading_Accuracy": accuracy_metrics['accuracy'],
            "Profit_Pct": accuracy_metrics['profit_pct'],
            "Buy_Signals": accuracy_metrics['buy_signals'],
            "Sell_Signals": accuracy_metrics['sell_signals']
        }
        
        if results_queue is not None:
            results_queue.put(result)
        return result
        
    except Exception as e:
        print(f"Error processing {stock}: {str(e)}")
        if results_queue is not None:
            results_queue.put({"Stock": stock, "Error": str(e)})
        return None

def process_stocks_parallel(stocks):
    """Process multiple stocks in parallel using multiprocessing."""
    # Create a manager for the results queue
    manager = mp.Manager()
    results_queue = manager.Queue()
    
    # Create a process pool
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        # Use partial to pass the results queue to each process
        process_func = partial(process_stock, results_queue=results_queue)
        
        # Process stocks in parallel
        pool.map(process_func, stocks)
    
    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    return [r for r in results if r is not None and 'Error' not in r]

if __name__ == "__main__":
    # Get list of stocks to process
    stocks = get_stocks_list()
    print(f"Starting processing of {len(stocks)} stocks using {NUM_PROCESSES} processes...")
    
    # Process stocks in parallel
    results = process_stocks_parallel(stocks)
    
    # Convert results to DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # Save results
        os.makedirs("results", exist_ok=True)
        results_df.to_csv("results/prophet_results.csv", index=False)
        
        # Print summary
        print("\n" + "="*50)
        print(f"Processed {len(results_df)} stocks successfully.")
        print(f"Average Trading Accuracy: {results_df['Trading_Accuracy'].mean()*100:.2f}%")
        print(f"Average Profit: ${results_df['Profit_Pct'].mean():.2f}%")
        print(f"Results saved to results/prophet_results.csv")
    else:
        print("No results to save.")
    