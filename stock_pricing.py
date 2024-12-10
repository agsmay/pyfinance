import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from trading_strategy import movingAverage

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

    
    def get_stock_data(self, save_to_csv : bool =False):
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

        # Create the stock_data DataFrame
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

        # If save_to_csv is True, save the stock data to a CSV file
        if save_to_csv:
            self.stock_data.to_csv(f"{self.ticker}_stock_data.csv")

    def plot_stock_price(self):
        """
        Plot historical stock price data.
        """

        self.stock_data["Close"].plot(title=f"{self.ticker} Stock Price", ylabel="Price (USD)")

if __name__ == "__main__":

    ticker = "AAPL"     # Ticker symbol for Apple Inc.
    period = "1y"       # Time period for fetching historical stock price data (1 year)

    # Period must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

    # Initialize the stock price object, fetch data for the specified period, and plot the stock price
    tesla = stockPrice(ticker)
    tesla.set_period(period)

    tesla.get_stock_data()
    # tesla.plot_stock_price()

    # Initialize and simulate the moving average trading strategy
    mov_average_strategy = movingAverage(tesla.ticker, tesla.stock_data)
    mov_average_strategy.simulate_strategy(short_window=20, long_window=50)

    # Plot the strategy results with buy and sell signals
    mov_average_strategy.plot_strategy()

    # Backtest the strategy with an initial balance of $10000 USD, then plot the results
    mov_average_strategy.backtest(initial_balance=10000)
    mov_average_strategy.plot_backtest()

    plt.show()
