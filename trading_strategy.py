import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import pandas_ta as ta

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


# import plotting

class tradeManager:
    """
    Trade manager object to run and backtest trading strategies.
    """

    def __init__(self, ticker : str, stock_data : pd.DataFrame):
        """
        Initialize the trade manager with a given ticker symbol and stock data."""
        self.ticker = ticker
        self.stock_data = stock_data
        self.strategies = {
            "Moving Average": movingAverage,
            "Hidden Markov Model": hiddenMarkov,
            "Donchian Channel": donchianChannel,
            "Decision Tree": decisionTreeTrading,
            "Kalman Trend": KalmanTrendStrategy,
            "Kalman Mean Reversion": KalmanMeanReversion,
            "Random Forest": randomForest,
            "Neural Network": NeuralNetworkStrategy,
            "Random Trader": randomTrader
        }

    def run_strategy(self, strategy_name : str, plot : bool, **kwargs):
        """
        Run a trading strategy with the given parameters.
        
        Inputs:
            strategy_name (str): The name of the trading strategy to run. Must be one of:
                - "Moving Average"
                - "Hidden Markov Model"
                - "Donchian Channel"
                - "Decision Tree"
                - "Kalman Trend"
                - "Kalman Mean Reversion"
                - "Random Forest"
                - "Neural Network"
            plot (bool): Whether to plot the trading strategy results.
            **kwargs: Additional keyword arguments for the trading strategy.
        """

        # Check if the strategy is valid
        self.strategy_name = strategy_name

        if self.strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found.")
        
        # Initialise and run the strategy
        StrategyClass = self.strategies[strategy_name]
        strategy = StrategyClass(self.ticker, self.stock_data)
        strategy.run_strategy(**kwargs)

        # Set the output of the strategy
        self.trade_data = strategy.set_output()

        # Plot the strategy results
        if plot:
            strategy.plot_strategy()
    
    def plot_backtest(self):
        """
        Plot the backtest results.

        Generates the following plots:
        - Portfolio value over time
        - Number of holdings over time
        - Cash balance over time
        """

        # Portfolio value plot
        plt.figure()
        plt.plot(self.trade_data.index, self.trade_data['Portfolio Value'], label='Portfolio Value', color='blue')
        plt.plot(self.trade_data.index, self.trade_data['Buy and Hold'], label='Buy and Hold', color='gray', linestyle='--')
        plt.title(f"{self.ticker} {self.strategy_name} Backtest")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (USD)")
        plt.legend()

        # Holdings plot
        plt.figure()
        plt.plot(self.trade_data.index, self.trade_data['Holdings'], label='Holdings', color='green')
        plt.title(f"{self.ticker} Holdings")
        plt.xlabel("Date")
        plt.ylabel("Number of Shares")
        plt.legend()

        # Cash balance plot
        plt.figure()
        plt.plot(self.trade_data.index, self.trade_data['Cash'], label='Cash', color='red')
        plt.title(f"{self.ticker} Cash Balance")
        plt.xlabel("Date")
        plt.ylabel("Cash (USD)")
        plt.legend()
    
    def buy_and_hold(self, initial_balance=10000):
        """
        Simulate a buy-and-hold strategy for comparison.
        """

        self.trade_data['Buy and Hold'] = self.trade_data['Close'] * (initial_balance / self.trade_data['Close'].iloc[0])

    def backtest_strategy(self, plot, initial_balance=10000):
        """
        Backtest the trading strategy by simulating trades for the given forecast period.
        
        Inputs:
            plot (bool): Whether to plot the backtest results.
            initial_balance (float): The initial cash balance for the backtest.
        """

        # Initialize trade data
        self.trade_data['Cash'] = initial_balance
        self.trade_data['Holdings'] = 0
        self.trade_data['Portfolio Value'] = initial_balance

        price_column = 'Close'

        # Loop through data and simulate trades
        for i in range(1, len(self.trade_data)):
            current_row = self.trade_data.index[i]
            prev_row = self.trade_data.index[i - 1]
            signal = self.trade_data.loc[current_row, 'Signal']
            prev_portfolio_value = self.trade_data.loc[prev_row, 'Portfolio Value']

            # If buy signal and enough cash, buy shares
            if signal == 1 and self.trade_data.loc[prev_row, 'Cash'] > 0:
                self.trade_data.loc[current_row, 'Holdings'] = (
                    prev_portfolio_value / self.trade_data.loc[current_row, price_column]
                )
                self.trade_data.loc[current_row, 'Cash'] = 0

            # If sell signal and enough shares, sell holdings
            elif signal == -1 and self.trade_data.loc[prev_row, 'Holdings'] > 0:
                self.trade_data.loc[current_row, 'Cash'] = (
                    self.trade_data.loc[prev_row, 'Holdings'] *
                    self.trade_data.loc[current_row, price_column]
                )
                self.trade_data.loc[current_row, 'Holdings'] = 0
            
            # Else, hold position
            else:
                self.trade_data.loc[current_row, 'Cash'] = self.trade_data.loc[prev_row, 'Cash']
                self.trade_data.loc[current_row, 'Holdings'] = self.trade_data.loc[prev_row, 'Holdings']

            # Update portfolio value
            self.trade_data.loc[current_row, 'Portfolio Value'] = (
                self.trade_data.loc[current_row, 'Cash'] +
                self.trade_data.loc[current_row, 'Holdings'] * self.trade_data.loc[current_row, price_column]
            )
        
        self.buy_and_hold(initial_balance=initial_balance)

        # Plot results of backtest
        if plot:
            self.plot_backtest()

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
    
    def run_strategy(self, lower_window=20, upper_window=50):
        """
        Run the moving average trading strategy.
        """
        self.add_moving_average(window=lower_window)
        self.add_moving_average(window=upper_window)
        self.add_rsi(window=14)

        lower_ma = f"MA_{lower_window}"
        upper_ma = f"MA_{upper_window}"

        # Generate buy/sell signals
        self.stock_data['Signal'] = 0

        self.stock_data.loc[
            (self.stock_data[lower_ma] > self.stock_data[upper_ma]) & 
            (self.stock_data['RSI'] < 30), 'Signal'] = 1
        
        self.stock_data.loc[
            (self.stock_data[lower_ma] < self.stock_data[upper_ma]) &
            (self.stock_data['RSI'] > 70), 'Signal'] = -1

    def set_output(self):
        """
        Set the output of the moving average strategy, with date as index, close price, signals."""
    
        self.strategy_df = self.stock_data[['Close', 'Signal']]
      
        return self.strategy_df

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

    def simulate_hmm_strategy(self):
        """
        Simulate a trading strategy based on HMM-predicted states.
        
        Inputs:
            state_to_action (dict): Mapping of HMM states to trading actions.
                                     Example: {0: 'hold', 1: 'buy', 2: 'sell'}
        """

        state_to_action = {
            'buy': 1,
            'sell': 2,
            'hold': 0
        }

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

    def run_strategy(self):
        """
        Run the HMM trading strategy with the given state-to-action mapping.
        """
        self.train_hmm()
        self.simulate_hmm_strategy()
    
    def set_output(self):
        """
        Set the output of the hidden Markov strategy, with date as index, close price, signals."""
    
        self.strategy_df = self.stock_data[['Close', 'Signal']]
      
        return self.strategy_df
    
    def plot_strategy(self):
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

    def run_strategy(self):
        """
        Simulate the Donchian Channel trading strategy given some initial balance.
        """

        self.calculate_donchian_channel()

        in_position = False
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
                buy_signals.append(i)
            
            elif stock_low < stock_dcl and in_position:
                sell_signals.append(i)

        self.stock_data['Buy Signal'] = np.nan
        self.stock_data['Sell Signal'] = np.nan
        self.stock_data.loc[buy_signals, 'Buy Signal'] = self.stock_data['Close']
        self.stock_data.loc[sell_signals, 'Sell Signal'] = self.stock_data['Close']

    def plot_strategy(self):
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

    def set_output(self):
        """
        Set the output of the Donchian channel strategy, with date as index, close price, signals."""
    
        self.strategy_df = self.stock_data[['Close', 'Signal']]
      
        return self.strategy_df

class decisionTreeTrading:
    def __init__(self, ticker : str, stock_data : pd.DataFrame):
        self.ticker = ticker
        self.stock_data = stock_data
        self.model = DecisionTreeClassifier()
        self.features = None  # Placeholder for feature matrix
        self.labels = None    # Placeholder for target labels

    def prepare_features(self):
        """
        Prepares features and labels for training the decision tree model.
        """
        # Example features: Moving averages and price changes
        self.stock_data['MA_5'] = self.stock_data['Close'].rolling(window=5).mean()
        self.stock_data['MA_10'] = self.stock_data['Close'].rolling(window=10).mean()
        self.stock_data['Price_Change'] = self.stock_data['Close'].pct_change()

        # Target labels: 1 (buy), 0 (hold), -1 (sell)
        self.stock_data['Signal'] = 0
        self.stock_data.loc[self.stock_data['Price_Change'] > 0.01, 'Signal'] = 1
        self.stock_data.loc[self.stock_data['Price_Change'] < -0.01, 'Signal'] = -1

        # Drop rows with NaN values
        self.stock_data.dropna(inplace=True)

        # Define features and labels
        self.features = self.stock_data[['MA_5', 'MA_10', 'Price_Change']].values
        self.labels = self.stock_data['Signal'].values

    def train_model(self):
        """
        Train the decision tree model.
        """
        if self.features is None or self.labels is None:
            self.prepare_features()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42
        )

        # Train the decision tree model
        self.model.fit(X_train, y_train)

        # Evaluate accuracy
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model trained. Accuracy: {accuracy:.2%}")

    def generate_signals(self):
        """
        Use the trained model to generate trading signals.
        """
        if self.features is None:
            self.prepare_features()

        self.stock_data['Signal'] = self.model.predict(self.features)

    def run_strategy(self):
        """
        Run the decision tree trading strategy.
        """
        self.train_model()
        self.generate_signals()

    def plot_strategy(self):
        """
        Plot the stock price with buy/sell signals.
        """
        buy_signals = self.stock_data[self.stock_data['Signal'] == 1]
        sell_signals = self.stock_data[self.stock_data['Signal'] == -1]

        plt.figure()
        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='black')
        plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green')
        plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red')
        plt.title(f"{self.ticker} Decision Tree Trading Strategy")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()

    def set_output(self):
        """
        Set the output of the decision tree strategy, with date as index, close price, signals."""
    
        self.strategy_df = self.stock_data[['Close', 'Signal']]
      
        return self.strategy_df
    
class KalmanTrendStrategy:
    def __init__(self, ticker : str, stock_data: pd.DataFrame):
        """
        Initialize the Kalman Trend Following Strategy.

        Parameters:
        stock_data (pd.DataFrame): A DataFrame containing stock price data with a 'Close' column.
        """
        self.ticker = ticker
        self.stock_data = stock_data
        self.trend = None
        self.velocity = None
        self.kalman_data = None

    def kalman_filter(self):
        """
        Apply Kalman filter to estimate the trend and velocity of the stock price.
        """
        # Initialize variables
        n = len(self.stock_data)
        observed_prices = self.stock_data['Close'].values

        # State vector [trend, velocity]
        x = np.zeros((2, n))
        x[:, 0] = [observed_prices[0], 0]  # Initial trend and velocity

        # State covariance
        P = np.eye(2)

        # Transition matrix
        F = np.array([[1, 1], [0, 1]])

        # Observation model
        H = np.array([[1, 0]])

        # Process noise covariance
        Q = np.array([[0.001, 0], [0, 0.001]])

        # Measurement noise covariance
        R = np.array([[1]])

        # Kalman filter loop
        for t in range(1, n):
            # Prediction step
            x_pred = F @ x[:, t-1]
            P_pred = F @ P @ F.T + Q

            # Update step
            y = observed_prices[t] - H @ x_pred  # Innovation
            S = H @ P_pred @ H.T + R  # Innovation covariance
            K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
            x[:, t] = x_pred + K.flatten() * y
            P = (np.eye(2) - K @ H) @ P_pred

        # Save results
        self.trend = x[0, :]
        self.velocity = x[1, :]
        self.kalman_data = pd.DataFrame({
            'Date': self.stock_data.index,
            'Observed': observed_prices,
            'Trend': self.trend,
            'Velocity': self.velocity
        })

    def generate_signals(self, threshold: float = 2.0):
        """
        Generate trading signals based on the Kalman filter trend.

        Parameters:
        threshold (float): The deviation threshold for generating signals.

        """
        deviations = self.kalman_data['Observed'] - self.kalman_data['Trend']
        self.kalman_data['Signal'] = 0
        self.kalman_data.loc[deviations > threshold, 'Signal'] = -1  # Sell signal
        self.kalman_data.loc[deviations < -threshold, 'Signal'] = 1  # Buy signal

    def plot_strategy(self):
        """
        Plot the observed price, estimated trend, and trading signals.
        """
        plt.figure()
        plt.plot(self.kalman_data['Date'], self.kalman_data['Observed'], label='Observed Price', color='blue')
        plt.plot(self.kalman_data['Date'], self.kalman_data['Trend'], label='Estimated Trend', color='orange')
        buy_signals = self.kalman_data[self.kalman_data['Signal'] == 1]
        sell_signals = self.kalman_data[self.kalman_data['Signal'] == -1]
        plt.scatter(buy_signals['Date'], buy_signals['Observed'], label='Buy Signal', color='green', marker='^', alpha=1)
        plt.scatter(sell_signals['Date'], sell_signals['Observed'], label='Sell Signal', color='red', marker='v', alpha=1)
        plt.title("Kalman Filter Trend Following Strategy")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()

    def run_strategy(self):
        self.kalman_filter()
        self.generate_signals(threshold=2.0)

    def set_output(self):
        """
        Set the output of the kalman trend strategy, with date as index, close price, signals."""
    
        self.strategy_df = self.kalman_data[['Observed', 'Signal']]
        self.strategy_df.rename(columns={'Observed': 'Close'}, inplace=True)

        return self.strategy_df
    
class KalmanMeanReversion:
    def __init__(self, ticker: str, stock_data: pd.DataFrame):
        """
        Initialize the Kalman Mean Reversion Strategy.

        Parameters:
        stock_data (pd.DataFrame): A DataFrame containing stock price data with a 'Close' column.
        """

        self.ticker = ticker
        self.stock_data = stock_data
        self.mean = None
        self.variance = None
        self.kalman_data = None

    def kalman_filter(self):
        """
        Apply Kalman filter to estimate the mean of the stock price.
        """
        n = len(self.stock_data)
        observed_prices = self.stock_data['Close'].values

        # Initialize Kalman filter parameters
        mean = np.zeros(n)
        variance = np.zeros(n)

        # Initial estimates
        mean[0] = observed_prices[0]
        variance[0] = 1

        # Process noise (controls smoothness)
        process_noise = 0.01

        # Measurement noise (controls sensitivity to new data)
        measurement_noise = 1

        # Kalman filter loop
        for t in range(1, n):
            # Prediction step
            predicted_mean = mean[t-1]
            predicted_variance = variance[t-1] + process_noise

            # Update step
            innovation = observed_prices[t] - predicted_mean
            innovation_variance = predicted_variance + measurement_noise
            kalman_gain = predicted_variance / innovation_variance

            mean[t] = predicted_mean + kalman_gain * innovation
            variance[t] = (1 - kalman_gain) * predicted_variance

        # Save results
        self.mean = mean
        self.variance = variance
        self.kalman_data = pd.DataFrame({
            'Date': self.stock_data.index,
            'Observed': observed_prices,
            'Estimated Mean': mean,
            'Variance': variance
        })

    def generate_signals(self, threshold: float = 1.5):
        """
        Generate trading signals based on the estimated mean.

        Parameters:
        threshold (float): The standard deviation multiplier for signal generation.
        """
        deviations = self.kalman_data['Observed'] - self.kalman_data['Estimated Mean']
        std_dev = np.sqrt(self.kalman_data['Variance'])

        self.kalman_data['Signal'] = 0
        self.kalman_data.loc[deviations > threshold * std_dev, 'Signal'] = -1  # Sell signal
        self.kalman_data.loc[deviations < -threshold * std_dev, 'Signal'] = 1  # Buy signal

    def plot_strategy(self):
        """
        Plot the observed price, estimated mean, and trading signals.
        """
        plt.figure()
        plt.plot(self.kalman_data['Date'], self.kalman_data['Observed'], label='Observed Price', color='blue')
        plt.plot(self.kalman_data['Date'], self.kalman_data['Estimated Mean'], label='Estimated Mean', color='orange')
        buy_signals = self.kalman_data[self.kalman_data['Signal'] == 1]
        sell_signals = self.kalman_data[self.kalman_data['Signal'] == -1]
        plt.scatter(buy_signals['Date'], buy_signals['Observed'], label='Buy Signal', color='green', marker='^', alpha=1)
        plt.scatter(sell_signals['Date'], sell_signals['Observed'], label='Sell Signal', color='red', marker='v', alpha=1)
        plt.title("Kalman Filter Mean Reversion Strategy")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()

    def run_strategy(self):

        self.kalman_filter()
        self.signals = self.generate_signals(threshold=1.5)
    
    def set_output(self):
        """
        Set the output of the kalman mean reversion strategy, with date as index, close price, signals."""
    
        self.strategy_df = self.kalman_data[['Observed', 'Signal']]
        self.strategy_df.rename(columns={'Observed': 'Close'}, inplace=True)

        return self.strategy_df

class randomForest:
    def __init__(self, ticker : str, stock_data : pd.DataFrame):
        """
        Initialize the trading strategy.

        Args:
            ticker (str): The stock ticker symbol.
            stock_data (pd.DataFrame): The stock data with Date as the index.
        """
        self.ticker = ticker
        self.stock_data = stock_data
        self.model = None

    def prepare_features(self):
        """
        Prepares features and labels for the classifier.
        """
        # Add technical indicators
        self.stock_data['SMA_10'] = self.stock_data['Close'].rolling(10).mean()
        self.stock_data['SMA_50'] = self.stock_data['Close'].rolling(50).mean()
        self.stock_data['RSI'] = self.calculate_rsi(self.stock_data['Close'])
        self.stock_data['Lag_1'] = self.stock_data['Close'].shift(1)
        self.stock_data['Lag_2'] = self.stock_data['Close'].shift(2)

        # Create binary labels based on future price movement
        self.stock_data['Target'] = np.sign(self.stock_data['Close'].diff().shift(-1))
        self.stock_data['Target'] = self.stock_data['Target'].replace(0, 0)  # Optional: Keep 'Hold' as 0

        # Drop rows with NaN values
        self.stock_data.dropna(inplace=True)

        # Features and target
        X = self.stock_data[['SMA_10', 'SMA_50', 'RSI', 'Lag_1', 'Lag_2']]
        y = self.stock_data['Target']

        return X, y

    @staticmethod
    def calculate_rsi(series, period=14):
        """
        Calculate Relative Strength Index (RSI).

        Args:
            series (pd.Series): Price series.
            period (int): Lookback period for RSI.
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def train_model(self):
        """
        Train a Random Forest Classifier on the data.
        """
        X, y = self.prepare_features()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)

    def generate_signals(self):
        """
        Generate trading signals (Buy, Sell, Hold) for the stock data.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train_model first.")

        X, _ = self.prepare_features()
        self.stock_data['Signal'] = self.model.predict(X)

    def run_strategy(self):
        """
        Run the Random Forest trading strategy.
        """
        self.train_model()
        self.generate_signals()
    
    def plot_strategy(self):
        """
        Plot the stock price with buy/sell signals.
        """
        buy_signals = self.stock_data[self.stock_data['Signal'] == 1]
        sell_signals = self.stock_data[self.stock_data['Signal'] == -1]

        plt.figure()
        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='black')
        plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green')
        plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red')
        plt.title(f"{self.ticker} Random Forest Trading Strategy")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()

    def set_output(self):
        """
        Set the output of the random forest strategy, with date as index, close price, signals."""
    
        self.strategy_df = self.stock_data[['Close', 'Signal']]
      
        return self.strategy_df

class NeuralNetworkStrategy:
    def __init__(self, ticker: str, stock_data: pd.DataFrame):
        self.ticker = ticker
        self.stock_data = stock_data
        self.model = None
        self.scaler = StandardScaler()

    def prepare_features(self):
        """
        Prepare features and labels for the neural network.
        """
        # Example features: Moving averages and RSI
        self.stock_data['SMA_10'] = self.stock_data['Close'].rolling(window=10).mean()
        self.stock_data['SMA_50'] = self.stock_data['Close'].rolling(window=50).mean()
        self.stock_data['RSI'] = self.calculate_rsi(self.stock_data['Close'])

        # Create a binary label: 1 for upward trend, 0 for downward trend
        self.stock_data['Target'] = np.where(self.stock_data['Close'].shift(-1) > self.stock_data['Close'], 1, 0)
        
        # Drop NaN rows after feature generation
        self.stock_data.dropna(inplace=True)

        # Features (X) and target (y)
        X = self.stock_data[['SMA_10', 'SMA_50', 'RSI']].values
        y = self.stock_data['Target'].values

        # Normalize features
        X = self.scaler.fit_transform(X)

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def build_model(self, input_dim):
        """
        Build the neural network model.
        """
        model = Sequential([
            Dense(64, input_dim=input_dim, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, epochs=50, batch_size=32):
        """
        Train the neural network model.
        """
        X_train, X_test, y_train, y_test = self.prepare_features()

        # Build the model
        self.model = self.build_model(input_dim=X_train.shape[1])

        # Train the model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    def predict_signals(self):
        """
        Predict trading signals using the trained model.
        """
        if not self.model:
            raise ValueError("Model is not trained. Call train_model first.")

        # Prepare features for prediction
        X = self.stock_data[['SMA_10', 'SMA_50', 'RSI']].values
        X = self.scaler.transform(X)

        # Predict probabilities
        predictions = self.model.predict(X)
        self.stock_data['Signal'] = np.where(predictions > 0.5, 1, -1)  # Buy (1) or Sell (-1)

    def plot_strategy(self):
        """
        Plot the stock price with buy, sell, and hold signals.
        """
        plt.figure()

        # Plot stock price
        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Stock Price', color='blue', linewidth=2)

        # Plot signals
        buy_signals = self.stock_data[self.stock_data['Signal'] == 1]
        sell_signals = self.stock_data[self.stock_data['Signal'] == -1]
        hold_signals = self.stock_data[self.stock_data['Signal'] == 0]

        # Use distinct markers for each signal
        plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green', s=100, alpha=0.7)
        plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red', s=100, alpha=0.7)
        plt.scatter(hold_signals.index, hold_signals['Close'], label='Hold Signal', marker='o', color='orange', s=60, alpha=0.5)

        # Enhance visualization
        plt.title(f"{self.ticker} Stock Price and Trading Signals", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Price (USD)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid()
    
    def run_strategy(self):
        """
        Run the neural network trading strategy.
        """
        self.train_model()
        self.predict_signals()

    def set_output(self):
        """
        Set the output of the neural network strategy, with date as index, close price, signals.
        """
    
        self.strategy_df = self.stock_data[['Close', 'Signal']]
      
        return self.strategy_df

class randomTrader:

    def __init__(self, ticker, stock_data):
        self.ticker = ticker
        self.stock_data = stock_data
    
    def run_strategy(self):
        """
        Run the random trading strategy.
        """
        self.stock_data['Signal'] = np.random.choice([-1, 0, 1], size=len(self.stock_data))
    
    def plot_strategy(self):
        """
        Plot the stock price with buy/sell signals.
        """
        buy_signals = self.stock_data[self.stock_data['Signal'] == 1]
        sell_signals = self.stock_data[self.stock_data['Signal'] == -1]

        plt.figure()
        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='black')
        plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green')
        plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red')
        plt.title(f"{self.ticker} Random Trading Strategy")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend

    def set_output(self):
        """
        Set the output of the neural network strategy, with date as index, close price, signals.
        """
    
        self.strategy_df = self.stock_data[['Close', 'Signal']]
      
        return self.strategy_df
