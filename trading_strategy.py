import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import requests
import pandas_ta as ta
from termcolor import colored as cl
import math
from alpaca_trade_api import REST, Stream
from transformers import pipeline

# import plotting

class movingAverage:
    """
    Moving average object to add moving averages to stock price data and simulate a trading strategy.
    """
    def __init__(self, ticker : str, stock_data : pd.DataFrame):
        """
        Initialize the moving average object with a given ticker symbol and stock data.
        
        Inputs:
            ticker (str): The ticker symbol for the stock.
            stock_data (pd.DataFrame): The historical stock price data.
        """
        self.ticker = ticker
        self.stock_data = stock_data

    def add_moving_average(self, window : int = 20):
        """
        Add a moving average to the stock data.

        Inputs:
            window (int): The window size for the moving average.
        """
        self.stock_data[f"MA_{window}"] = self.stock_data["Close"].rolling(window=window).mean()
    
    def add_rsi(self, window : int = 14):
        """
        Add the Relative Strength Index (RSI) to the stock data.

        Inputs:
            window (int): The window size for RSI calculation.
        """
        delta = self.stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.stock_data['RSI'] = 100 - (100 / (1 + rs))

    def plot_with_indicators(self):
        """
        Plot stock price with moving averages.
        """

        self.stock_data[["Close", "MA_20", "MA_50"]].plot(title=f"{self.ticker} with Indicators")
        plt.ylabel("Price (USD)")


    def plot_rsi(self):
        """
        Plot the Relative Strength Index (RSI).
        """

        self.stock_data["RSI"].plot(title=f"{self.ticker} RSI", ylabel="RSI")

    def simulate_strategy(self, short_window : int = 20, long_window : int = 50):
        """
        Simulate a moving average crossover strategy.

        Inputs:
            short_window (int): The window size for the short moving average.
            long_window (int): The window size for the long moving average.
        """

        # Add moving averages
        self.add_moving_average(window=short_window)
        self.add_moving_average(window=long_window)

        # Drop rows with NaN values
        self.stock_data.dropna(inplace=True)

        # Generate signals
        self.stock_data['Signal'] = 0

        self.stock_data.loc[self.stock_data[f"MA_{short_window}"] > self.stock_data[f"MA_{long_window}"], 'Signal'] = 1 # Buy signal
        self.stock_data.loc[self.stock_data[f"MA_{short_window}"] < self.stock_data[f"MA_{long_window}"], 'Signal'] = -1    # Sell signal

        # Create position column
        self.stock_data['Position'] = self.stock_data['Signal'].shift()

        # Calculate returns
        self.stock_data['Daily Return'] = self.stock_data['Close'].pct_change()
        self.stock_data['Strategy Return'] = self.stock_data['Daily Return'] * self.stock_data['Position']

    def backtest(self, initial_balance : float = 10000.0):
        """
        Backtest the trading strategy. At the moment buy signals mean all the cash is converted to holdings and sell signals mean all the holdings are converted to cash.

        Inputs:
            initial_balance (float): Starting amount of money.
        """

        # Initialize Cash, Holdings, and Portfolio Value columns
        self.stock_data['Cash'] = initial_balance
        self.stock_data['Holdings'] = 0
        self.stock_data['Portfolio Value'] = initial_balance

        # Loop through the stock data
        for i in range(1, len(self.stock_data)):
            current_row = self.stock_data.index[i]
            prev_row = self.stock_data.index[i - 1]

            # Get signal for the current day
            signal = self.stock_data.loc[current_row, 'Signal']
            prev_portfolio_value = self.stock_data.loc[prev_row, 'Portfolio Value']

            if signal == 1 and self.stock_data.loc[prev_row, 'Cash'] > 0:  # Buy signal
                # Buy as much as possible with available portfolio value
                self.stock_data.loc[current_row, 'Holdings'] = (
                    prev_portfolio_value / self.stock_data.loc[current_row, 'Close']
                )
                self.stock_data.loc[current_row, 'Cash'] = 0

            elif signal == -1 and self.stock_data.loc[prev_row, 'Holdings'] > 0:  # Sell signal (only if we have holdings)
                # Sell all holdings and convert to cash
                self.stock_data.loc[current_row, 'Cash'] = (
                    self.stock_data.loc[prev_row, 'Holdings'] * self.stock_data.loc[current_row, 'Close']
                )
                self.stock_data.loc[current_row, 'Holdings'] = 0

            else:  # Hold (no action)
                self.stock_data.loc[current_row, 'Cash'] = self.stock_data.loc[prev_row, 'Cash']
                self.stock_data.loc[current_row, 'Holdings'] = self.stock_data.loc[prev_row, 'Holdings']

            # Update portfolio value for the current row
            self.stock_data.loc[current_row, 'Portfolio Value'] = (
                self.stock_data.loc[current_row, 'Cash'] +
                self.stock_data.loc[current_row, 'Holdings'] * self.stock_data.loc[current_row, 'Close']
            )

    def plot_backtest(self):
        """
        Plot the portfolio value over time.
        """

        plt.figure()
        plt.plot(self.stock_data.index.to_numpy(), self.stock_data['Portfolio Value'].to_numpy(), label='Portfolio Value')
        plt.title(f"{self.ticker} Trading Strategy Backtest")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (USD)")
        plt.legend()

    def plot_strategy(self):
        """
        Plot stock prices with buy/sell signals for the trading strategy.
        """

        buy_signals = self.stock_data.loc[self.stock_data['Signal'] == 1]
        sell_signals = self.stock_data.loc[self.stock_data['Signal'] == -1]

        plt.figure()
        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', alpha=0.7)

        # Buy signals
        plt.scatter(buy_signals.index, buy_signals['Close'], 
                    label='Buy Signal', marker='^', color='green')

        # Sell signals
        plt.scatter(sell_signals.index, sell_signals['Close'], 
                    label='Sell Signal', marker='v', color='red')

        plt.title(f"{self.ticker} Trading Strategy")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()

    def backtest(self, initial_balance: float = 10000.0, use_predictions=False):
        """
        Backtest the strategy on true or predicted prices.

        Inputs:
            initial_balance (float): Starting portfolio value.
            use_predictions (bool): Whether to use predicted prices.
        """
        self.stock_data['Cash'] = initial_balance
        self.stock_data['Holdings'] = 0
        self.stock_data['Portfolio Value'] = initial_balance

        price_column = 'Predicted Close' if use_predictions else 'Close'

        for i in range(1, len(self.stock_data)):
            current_row = self.stock_data.index[i]
            prev_row = self.stock_data.index[i - 1]
            signal = self.stock_data.loc[current_row, 'Signal']
            prev_portfolio_value = self.stock_data.loc[prev_row, 'Portfolio Value']

            if signal == 1 and self.stock_data.loc[prev_row, 'Cash'] > 0:
                self.stock_data.loc[current_row, 'Holdings'] = (
                    prev_portfolio_value / self.stock_data.loc[current_row, price_column]
                )
                self.stock_data.loc[current_row, 'Cash'] = 0
            elif signal == -1 and self.stock_data.loc[prev_row, 'Holdings'] > 0:
                self.stock_data.loc[current_row, 'Cash'] = (
                    self.stock_data.loc[prev_row, 'Holdings'] *
                    self.stock_data.loc[current_row, price_column]
                )
                self.stock_data.loc[current_row, 'Holdings'] = 0
            else:
                self.stock_data.loc[current_row, 'Cash'] = self.stock_data.loc[prev_row, 'Cash']
                self.stock_data.loc[current_row, 'Holdings'] = self.stock_data.loc[prev_row, 'Holdings']

            self.stock_data.loc[current_row, 'Portfolio Value'] = (
                self.stock_data.loc[current_row, 'Cash'] +
                self.stock_data.loc[current_row, 'Holdings'] * self.stock_data.loc[current_row, price_column]
            )

    def compare_backtests(self):
        """
        Compare portfolio values on true vs. predicted prices.
        """
        self.backtest(initial_balance=10000.0, use_predictions=False)
        self.stock_data['Portfolio Value True'] = self.stock_data['Portfolio Value']

        self.backtest(initial_balance=10000.0, use_predictions=True)
        self.stock_data['Portfolio Value Predicted'] = self.stock_data['Portfolio Value']

        plt.figure()
        plt.plot(self.stock_data.index, self.stock_data['Portfolio Value True'], label='True Portfolio Value', alpha=0.7)
        plt.plot(self.stock_data.index, self.stock_data['Portfolio Value Predicted'], label='Predicted Portfolio Value', alpha=0.7, linestyle='--')
        plt.title(f"{self.ticker} Portfolio Value: True vs Predicted")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (USD)")
        plt.legend()

class hiddenMarkov:
    def __init__(self, ticker : str, stock_data : pd.DataFrame):
        self.ticker = ticker
        self.stock_data = stock_data

    def train_hmm(self, n_states : int = 3, features : list = ['Daily Return']):
        """
        Train a Hidden Markov Model (HMM) on the stock data.
        
        Inputs:
            n_states (int): Number of hidden states.
            features (list): List of feature columns to use for HMM training.
        """
        # Extract the features for training
        X = self.stock_data[features].dropna().to_numpy()
        
        # Train the HMM
        self.hmm = GaussianHMM(n_components=n_states, covariance_type="full", random_state=42)
        self.hmm.fit(X)
        
        # Decode the hidden states
        hidden_states = self.hmm.predict(X)
        self.stock_data['HMM State'] = np.nan
        self.stock_data.loc[self.stock_data[features].dropna().index, 'HMM State'] = hidden_states

    def simulate_hmm_strategy(self, state_to_action : dict):
        """
        Simulate a trading strategy based on HMM-predicted states.
        
        Inputs:
            state_to_action (dict): Mapping of HMM states to trading actions.
                                     Example: {0: 'hold', 1: 'buy', 2: 'sell'}
        """
        # Ensure the HMM has been trained
        if 'HMM State' not in self.stock_data.columns:
            raise ValueError("HMM not trained. Call train_hmm() first.")
        
        # Initialize signals based on HMM states
        self.stock_data['Signal'] = 0  # Default to hold
        for state, action in state_to_action.items():
            if action == 'buy':
                self.stock_data.loc[self.stock_data['HMM State'] == state, 'Signal'] = 1
            elif action == 'sell':
                self.stock_data.loc[self.stock_data['HMM State'] == state, 'Signal'] = -1

        # Use the backtesting method to evaluate the strategy
        self.backtest()

    def plot_hmm_states(self):
        """
        Plot stock prices with the hidden states overlayed.
        """
        plt.figure()
        for state in self.stock_data['HMM State'].dropna().unique():
            state_data = self.stock_data[self.stock_data['HMM State'] == state]
            plt.scatter(state_data.index, state_data['Close'], label=f'State {int(state)}', alpha=0.6)

        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='black', linewidth=0.5)
        plt.title(f"{self.ticker} Hidden States")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()

    def backtest(self, initial_balance: float = 10000.0, use_predictions : bool = False):
        """
        Backtest the strategy on true or predicted prices.

        Inputs:
            initial_balance (float): Starting portfolio value.
            use_predictions (bool): Whether to use predicted prices.
        """
        self.stock_data['Cash'] = initial_balance
        self.stock_data['Holdings'] = 0
        self.stock_data['Portfolio Value'] = initial_balance

        price_column = 'Predicted Close' if use_predictions else 'Close'

        for i in range(1, len(self.stock_data)):
            current_row = self.stock_data.index[i]
            prev_row = self.stock_data.index[i - 1]
            signal = self.stock_data.loc[current_row, 'Signal']
            prev_portfolio_value = self.stock_data.loc[prev_row, 'Portfolio Value']

            if signal == 1 and self.stock_data.loc[prev_row, 'Cash'] > 0:
                self.stock_data.loc[current_row, 'Holdings'] = (
                    prev_portfolio_value / self.stock_data.loc[current_row, price_column]
                )
                self.stock_data.loc[current_row, 'Cash'] = 0
            elif signal == -1 and self.stock_data.loc[prev_row, 'Holdings'] > 0:
                self.stock_data.loc[current_row, 'Cash'] = (
                    self.stock_data.loc[prev_row, 'Holdings'] *
                    self.stock_data.loc[current_row, price_column]
                )
                self.stock_data.loc[current_row, 'Holdings'] = 0
            else:
                self.stock_data.loc[current_row, 'Cash'] = self.stock_data.loc[prev_row, 'Cash']
                self.stock_data.loc[current_row, 'Holdings'] = self.stock_data.loc[prev_row, 'Holdings']

            self.stock_data.loc[current_row, 'Portfolio Value'] = (
                self.stock_data.loc[current_row, 'Cash'] +
                self.stock_data.loc[current_row, 'Holdings'] * self.stock_data.loc[current_row, price_column]
            )

    def plot_backtest(self):
        """
        Plot the portfolio value over time.
        """

        plt.figure()
        plt.plot(self.stock_data.index.to_numpy(), self.stock_data['Portfolio Value'].to_numpy(), label='Portfolio Value')
        plt.title(f"{self.ticker} Trading Strategy Backtest")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (USD)")
        plt.legend()

class donchianChannel:
    """ Donchian Channel trading strategy.
    """

    def __init__(self, ticker : str, stock_data : pd.DataFrame):
        """
        Initialize the Donchian Channel strategy with a given ticker symbol and stock data.
        """

        self.ticker = ticker
        self.stock_data = stock_data

    def calculate_donchian_channel(self, lower_length = 40, upper_length = 50):
        """
        Calculate the Donchian Channel values for the stock data.
        """

        self.stock_data[['dcl', 'dcm', 'dcu']] = self.stock_data.ta.donchian(lower_length, upper_length)

    def plot_donchian_channel(self):
        """ 
        Plot the stock price with the Donchian Channel.
        """

        plt.figure()
        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='red')
        plt.plot(self.stock_data.index, self.stock_data['dcl'], label='dcl', color='black', linestyle='--', alpha=0.3)
        plt.plot(self.stock_data.index, self.stock_data['dcm'], label='dcm', color='blue')
        plt.plot(self.stock_data.index, self.stock_data['dcu'], label='dcu', color='black', linestyle='--', alpha=0.3)
        plt.title(f"{self.ticker} Donchian Channel")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()

    def simulate_strategy(self, initial_balance=10000):
        """
        Simulate the Donchian Channel trading strategy given some initial balance.
        """

        self.calculate_donchian_channel()

        in_position = False
        equity = initial_balance
        buy_signals = []
        sell_signals = []

        for j in range(3, len(self.stock_data)):

            i = self.stock_data.index[j]

            stock_high = self.stock_data.loc[i, 'High']
            stock_close = self.stock_data.loc[i, 'Close']
            stock_low = self.stock_data.loc[i, 'Low']
            stock_dcu = self.stock_data.loc[i, 'dcu']
            stock_dcl = self.stock_data.loc[i, 'dcl']

            if stock_high > stock_dcu and not in_position:
                no_of_shares = math.floor(equity/stock_close)
                equity -= (no_of_shares * stock_close)
                in_position = True
                buy_signals.append(i)
            
            elif stock_low < stock_dcl and in_position:
                equity += (no_of_shares * stock_close)
                in_position = False
                sell_signals.append(i)

        if in_position:
            equity += (no_of_shares * stock_close)

        earning = round(equity - initial_balance, 2)
        roi = round((earning/initial_balance) * 100, 2)
        print(cl(f'EARNING: ${earning} ; ROI: {roi}%', attrs = ['bold']))

        self.stock_data['Buy Signal'] = np.nan
        self.stock_data['Sell Signal'] = np.nan
        self.stock_data.loc[buy_signals, 'Buy Signal'] = self.stock_data['Close']
        self.stock_data.loc[sell_signals, 'Sell Signal'] = self.stock_data['Close']

    def plot_donchian_channel(self):
        """ 
        Plot the stock price with the Donchian Channel and buy/sell signals.
        """

        plt.figure()
        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='red')
        plt.plot(self.stock_data.index, self.stock_data['dcl'], label='dcl', color='black', linestyle='--', alpha=0.3)
        plt.plot(self.stock_data.index, self.stock_data['dcm'], label='dcm', color='blue')
        plt.plot(self.stock_data.index, self.stock_data['dcu'], label='dcu', color='black', linestyle='--', alpha=0.3)
        
        # Plot buy/sell signals
        plt.scatter(self.stock_data.index, self.stock_data['Buy Signal'], label='Buy Signal', marker='^', color='green')
        plt.scatter(self.stock_data.index, self.stock_data['Sell Signal'], label='Sell Signal', marker='v', color='red')

        plt.title(f"{self.ticker} Donchian Channel")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
