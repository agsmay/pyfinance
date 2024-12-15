import pandas as pd
import matplotlib.pyplot as plt

import trading_strategy as ts  # Import all trading strategies
from stock_pricing import stockPrice

class StockPortfolio:
    def __init__(self, initial_cash=100000):
        """
        Initialize the StockPortfolio class.
        
        Args:
            initial_cash (float): Starting cash for the portfolio.
        """
        self.companies = {}  # {ticker: stockPrice instance}
        self.strategies = {}  # {ticker: {strategy_name: strategy_instance}}
        self.portfolio = {
            'cash': initial_cash,
            'positions': {},  # {ticker: number_of_shares}
            'history': []  # Transaction history
        }
        self.performance = pd.DataFrame()  # To track overall performance

    def add_company(self, ticker, stock_data):
        """
        Add a company to the portfolio.
        
        Args:
            ticker (str): The ticker symbol of the company.
            stock_data (pd.DataFrame): Historical stock data.
        """
        self.companies[ticker] = stockPrice(ticker)
        self.companies[ticker].stock_data = stock_data

    def assign_strategy(self, ticker, strategy_name, strategy_class, **kwargs):
        """
        Assign a strategy to a specific company.
        
        Args:
            ticker (str): The ticker symbol.
            strategy_name (str): A name for the strategy.
            strategy_class (class): The class of the strategy to instantiate.
            **kwargs: Additional arguments to pass to the strategy class.
        """
        if ticker not in self.strategies:
            self.strategies[ticker] = {}
        
        # Initialize strategy instance with common parameters
        strategy_instance = strategy_class(ticker=ticker, stock_data=self.companies[ticker].stock_data)

        # Run the strategy with strategy-specific parameters
        if hasattr(strategy_instance, "run_strategy"):
            strategy_instance.run_strategy(**kwargs)
        else:
            raise AttributeError(f"{strategy_name} does not have a run_strategy method.")
        self.strategies[ticker][strategy_name] = strategy_instance

    def run_strategies(self):
        """
        Run all assigned strategies for each company and collect outputs.
        """
        for ticker, strategy_dict in self.strategies.items():
            for strategy_name, strategy in strategy_dict.items():
                print(f"Running {strategy_name} for {ticker}")
                strategy.run_strategy()

    def backtest_portfolio(self):
        """
        Backtest the portfolio using strategy outputs.
        """
        for ticker, strategy_dict in self.strategies.items():
            for strategy_name, strategy in strategy_dict.items():
                output = strategy.strategy_df
                print(f"Backtesting {strategy_name} for {ticker}")
                # Perform backtesting logic here (e.g., simulate trades based on signals)

    def evaluate_performance(self):
        """
        Evaluate and store portfolio performance.
        """
        # Combine performance metrics from each company
        for ticker in self.companies:
            # Example: calculate returns and add to performance DataFrame
            company_data = self.companies[ticker].stock_data
            returns = company_data['Close'].pct_change().fillna(0)
            self.performance[ticker] = returns
        
        self.performance['Portfolio Value'] = self.performance.mean(axis=1)

    def visualise_performance(self):
        """
        Visualize portfolio performance with separate plots for clarity.
        """

        # 1. Portfolio Value Over Time
        plt.figure(figsize=(12, 6))
        plt.plot(self.performance.index, self.performance['Portfolio Value'], label="Portfolio Value", color="blue", linewidth=2)
        plt.title("Portfolio Value Over Time")
        plt.ylabel("Portfolio Value")
        plt.xlabel("Date")
        plt.grid()
        plt.legend()

        # 2. Stock Prices with Buy/Sell Signals
        for ticker in self.companies:
            stock_data = self.companies[ticker].stock_data
            if 'Signal' in stock_data.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(stock_data.index, stock_data['Close'], label=f"{ticker} Stock Price")
                buy_signals = stock_data[stock_data['Signal'] == 1]
                sell_signals = stock_data[stock_data['Signal'] == -1]
                plt.scatter(buy_signals.index, buy_signals['Close'], label=f"{ticker} Buy Signal", marker="^", color="green", alpha=0.6)
                plt.scatter(sell_signals.index, sell_signals['Close'], label=f"{ticker} Sell Signal", marker="v", color="red", alpha=0.6)
                plt.title(f"Stock Prices with Buy/Sell Signals for {ticker}")
                plt.ylabel("Price")
                plt.xlabel("Date")
                plt.grid()
                plt.legend()

        # 3. Cash Balance Over Time
        if 'Cash' in self.performance.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(self.performance.index, self.performance['Cash'], label="Cash Balance", color="orange", linestyle="--")
            plt.title("Cash Balance Over Time")
            plt.ylabel("Cash")
            plt.xlabel("Date")
            plt.grid()
            plt.legend()

        # Cumulative Returns
        if 'Portfolio Value' in self.performance.columns:
            self.performance['Cumulative Returns'] = (
                self.performance['Portfolio Value'] / self.performance['Portfolio Value'].iloc[0]
            ) - 1  # Normalize to start from 0
            plt.figure(figsize=(12, 6))
            plt.plot(self.performance.index, self.performance['Cumulative Returns'], label="Cumulative Returns", color="purple")
            plt.title("Cumulative Returns")
            plt.ylabel("Cumulative Returns")
            plt.xlabel("Date")
            plt.grid()
            plt.legend()

        # 5. Drawdowns Over Time
        self.performance['Drawdown'] = self.performance['Portfolio Value'] / self.performance['Portfolio Value'].cummax() - 1
        plt.figure(figsize=(12, 6))
        plt.plot(self.performance.index, self.performance['Drawdown'], label="Drawdown", color="red")
        plt.title("Drawdowns Over Time")
        plt.ylabel("Drawdown")
        plt.xlabel("Date")
        plt.grid()
        plt.legend()
