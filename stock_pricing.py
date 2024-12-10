import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from trading_strategy import movingAverage


class stockPrice:
    def __init__(self, ticker : str):
        self.ticker = ticker

    def set_period(self, period : str):
        self.period = period

    
    def get_stock_data(self, save_to_csv : bool =False):
        """
        Fetch historical stock price data for a given ticker symbol.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., "AAPL").
            save_to_csv (bool): Whether to save data to a CSV file.
            
        """

        # Use this to avoid yf.download() ridiculous data formats
        self.ticker_obj = yf.Ticker(self.ticker)
        self.historical_data = self.ticker_obj.history(self.period)

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
    
        if save_to_csv:
            self.stock_data.to_csv(f"{self.ticker}_stock_data.csv")

    def plot_stock_price(self):
        """
        Plot historical stock price data.
        """
        self.stock_data["Close"].plot(title=f"{self.ticker} Stock Price", ylabel="Price (USD)")



if __name__ == "__main__":

    ticker = "AAPL"
    period = "1y"

    # Period must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

    # Initialize and fetch data
    tesla = stockPrice(ticker)
    tesla.set_period(period)

    tesla.get_stock_data()
    tesla.plot_stock_price()

    # Initialize the moving average strategy
    mov_average_strategy = movingAverage(tesla.ticker, tesla.stock_data)
    mov_average_strategy.simulate_strategy(short_window=20, long_window=50)

    mov_average_strategy.plot_strategy()

    mov_average_strategy.backtest(initial_balance=10000)
    mov_average_strategy.plot_backtest()

    plt.show()


