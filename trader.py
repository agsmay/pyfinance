from stock_pricing import stockPrice
import trading_strategy as ts
from stock_portfolio import StockPortfolio  # Import the portfolio class
import matplotlib.pyplot as plt

# Settings for the portfolio
initial_cash = 100000  # Starting cash for the portfolio
companies = [("UNH", "1y"), ("GOGL", "1y"), ("AAPL", "1y")]  # List of (ticker, period) tuples
strategies = [
    ("UNH", "Moving Average", ts.movingAverage, {"lower_window": 20, "upper_window": 50}),
    ("GOGL", "Random Forest", ts.randomForest, {}),
    ("AAPL", "Neural Network", ts.NeuralNetworkStrategy, {}),
    ("AAPL", "Kalman Mean Reversion", ts.KalmanMeanReversion, {})
]


# Initialize the portfolio
portfolio = StockPortfolio(initial_cash=initial_cash)

# Add companies and fetch their stock data
for ticker, period in companies:
    company = stockPrice(ticker)
    company.set_period(period)
    company.get_stock_data()
    portfolio.add_company(ticker, company.stock_data)

# Assign strategies to companies
for ticker, strategy_name, strategy_class, kwargs in strategies:
    portfolio.assign_strategy(ticker, strategy_name, strategy_class, **kwargs)

# Run all assigned strategies
portfolio.run_strategies()

# Backtest the portfolio
portfolio.backtest_portfolio()

# Evaluate and visualize portfolio performance
portfolio.evaluate_performance()
portfolio.visualise_performance()

# Show all plots
plt.show()
