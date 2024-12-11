import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import trading_strategy as ts

class stockPrice:
    """ 
    Stock price object to fetch and analyze historical stock price data.
    """
    def __init__(self, ticker : str):
        """
        Initialize the stock price object with a given ticker symbol (e.g., "AAPL").

        Inputs:
            ticker (str): The ticker symbol for the stock.
        """

        self.ticker = ticker

    def set_period(self, period : str):
        """
        Set the period for fetching historical stock price data.

        Inputs:
            period (str): The period for fetching historical stock price data.
            Must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'].
        """

        self.period = period

    
    def get_stock_data(self, save_to_csv : bool = False):
        """
        Fetch historical stock price data for a given ticker symbol.
        
        Inputs:
            save_to_csv (bool): Whether to save data to a CSV file.
            
        """

        # Looks dumb but use this to avoid yf.download() ridiculous data formats
        self.ticker_obj = yf.Ticker(self.ticker)
        self.historical_data = self.ticker_obj.history(self.period)

        # Extract all relevant data
        dates = self.historical_data.index.to_numpy()
        open_prices = self.historical_data["Open"].to_numpy()
        high_prices = self.historical_data["High"].to_numpy()
        low_prices = self.historical_data["Low"].to_numpy()
        close_prices = self.historical_data["Close"].to_numpy()
        volumes = self.historical_data["Volume"].to_numpy()
        dividends = self.historical_data["Dividends"].to_numpy()
        stock_splits = self.historical_data["Stock Splits"].to_numpy()

        self.stock_data = pd.DataFrame({
            "Date": dates,
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volumes,
            "Dividends": dividends,
            "Stock Splits": stock_splits
        })
        self.stock_data.index = pd.to_datetime(self.stock_data['Date'])
        self.stock_data.drop(columns=['Date'], inplace=True)


        # If save_to_csv is True, save the stock data to a CSV file
        if save_to_csv:
            self.stock_data.to_csv(f"{self.ticker}_stock_data.csv")

    def plot_stock_price(self):
        """
        Plot historical stock price data.
        """

        self.stock_data["Close"].plot(title=f"{self.ticker} Stock Price", ylabel="Price (USD)")

    def prepare_features(self):
        """
        Prepares feature and target variables for the regression model.
        """
        # Create lagging features (previous close prices)
        self.stock_data['Lag_1'] = self.stock_data['Close'].shift(1)
        self.stock_data['Lag_2'] = self.stock_data['Close'].shift(2)
        self.stock_data.dropna(inplace=True)

        # Features: Lagged prices
        X = self.stock_data[['Lag_1', 'Lag_2']]

        # Target: Current close price
        y = self.stock_data['Close']
        return X, y

    def train_regression_model(self):
        """
        Train a regression model to predict stock prices.
        """
        # Prepare lagging features
        X, y = self.prepare_features()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Store predictions in the dataframe
        self.stock_data['Predicted Close'] = self.model.predict(X)

        # Evaluate the model
        self.mse = mean_squared_error(y_test, self.model.predict(X_test))
        print(f"Regression model trained. Test MSE: {self.mse:.4f}")

    def forecast_future_prices(self, forecast_days: int = 10):
        """
        Forecast future stock prices based on the trained regression model.

        Inputs:
            forecast_days (int): Number of days to forecast into the future.
        """
        if not hasattr(self, 'model') or not self.model:
            raise ValueError("Model is not trained. Call train_regression_model first.")

        self.future_forecast = []
        future_dates = []

        # Start from the last available date in the historical data
        last_date = self.stock_data.index[-1]
        last_features = self.stock_data[['Lag_1', 'Lag_2']].iloc[-1].values

        for _ in range(forecast_days):
            # Predict the next price using the current lagged features
            next_price = self.model.predict([last_features])[0]
            self.future_forecast.append(next_price)

            # Increment the date (skip weekends)
            next_date = last_date + pd.Timedelta(days=1)
            while next_date.weekday() >= 5:  # Skip weekends
                next_date += pd.Timedelta(days=1)
            future_dates.append(next_date)

            # Update the lagged features for the next iteration
            last_features = [next_price, last_features[0]]

            # Create a new row for the future prediction
            new_row = {
                'Lag_1': last_features[0],
                'Lag_2': last_features[1],
                'Close': None,  # No actual Close value for forecasted dates
                'Predicted Close': None,  # Placeholder, not used for future forecasts
            }

            # Add the new row to the DataFrame
            new_df = pd.DataFrame(new_row, index=[next_date])
            self.stock_data = pd.concat([self.stock_data, new_df])

            last_date = next_date

        # Create a separate DataFrame for the forecasted prices
        self.forecast_df = pd.DataFrame({
            'Forecasted Close': self.future_forecast
        }, index=future_dates)

        print("Future prices forecasted.")




    def predict_next_price(self):
        """
        Predicts the next day's stock price based on the trained model.
        """

        if not self.model:
            raise ValueError("Model is not trained. Call train_regression_model first.")

        # Use the most recent lagged features
        latest_data = self.stock_data.iloc[-2:][['Close']].values.flatten()
        lagged_features = np.array([latest_data[-1], latest_data[-2]]).reshape(1, -1)

        next_price = self.model.predict(lagged_features)[0]
        print(f"Predicted next price: {next_price:.2f}")
        return next_price

    def plot_regression_results(self):
        """
        Plot the regression results for historical stock prices.
        """
        X, y = self.prepare_features()

        # Ensure model is trained before plotting
        if not hasattr(self, 'model'):
            raise ValueError("Model not trained. Call 'train_regression_model' first.")

        # Plot the actual and predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Actual Price')
        plt.plot(self.stock_data.index, self.stock_data['Predicted Close'], label='Predicted Price', linestyle='--')
        plt.title(f"{self.ticker} Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()

    def run_regression_prediction(self, forecast_days: int = 10):
        """
        Run the full regression prediction pipeline: train model, forecast future prices, and plot results.

        Inputs:
            forecast_days (int): Number of days to forecast into the future.
        """
        # Train the regression model
        self.train_regression_model()

        # Forecast future prices
        self.forecast_future_prices(forecast_days=forecast_days)

        # Plot forecast and regression results
        self.plot_forecast()

    def plot_forecast(self):
        """
        Plot the predicted and forecasted stock prices, combining historical and future data.
        """
        plt.figure(figsize=(14, 7))

        # Plot historical and forecasted data
        historical_close = self.stock_data['Close'].dropna()
        predicted_close = self.stock_data['Predicted Close'].dropna()
        forecasted_close = self.forecast_df['Forecasted Close'].dropna()

        plt.plot(historical_close.index, historical_close, label='Historical Close')
        plt.plot(predicted_close.index, predicted_close, label='Predicted Close', linestyle='--')
        plt.plot(forecasted_close.index, forecasted_close, label='Forecasted Future', linestyle=':')

        # Add a vertical dashed line at today's date
        today = pd.Timestamp.today()
        plt.axvline(x=today, color='k', linestyle='--', label="Today's Date")

        plt.title(f"{self.ticker} Stock Price Prediction and Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()



if __name__ == "__main__":

    ticker = "TSLA"     # Ticker symbol
    period = "1y"       # Time period for fetching historical stock price data

    forecast_days = 10  # Number of days to forecast into the future

    # Period must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

    # Initialize the stock price object, fetch data for the specified period, and plot the stock price
    tesla = stockPrice(ticker)
    tesla.set_period(period)

    tesla.get_stock_data()
    # tesla.plot_stock_price()

    tesla.run_regression_prediction(forecast_days=forecast_days)

    # Initialize and simulate the moving average trading strategy
    mov_average_strategy = ts.movingAverage(tesla.ticker, tesla.stock_data)
    mov_average_strategy.simulate_strategy(short_window=20, long_window=50)

    mov_average_strategy.backtest()
    mov_average_strategy.plot_backtest()

    # Initialize Hidden Markov Model strategy
    hmm_strategy = ts.hiddenMarkov(ticker="TSLA", stock_data=tesla.stock_data)

    # Preprocess data and add daily returns
    hmm_strategy.stock_data['Daily Return'] = hmm_strategy.stock_data['Close'].pct_change()

    # Train HMM
    hmm_strategy.train_hmm(n_states=3, features=['Daily Return'])

    # Map states to actions
    state_to_action = {
        0: 'hold',  # Neutral
        1: 'buy',   # Bullish
        2: 'sell'   # Bearish
    }

    # Simulate HMM-based strategy
    hmm_strategy.simulate_hmm_strategy(state_to_action=state_to_action)

    # Plot hidden states and backtesting results
    hmm_strategy.plot_hmm_states()
    hmm_strategy.plot_backtest()


    plt.show()
