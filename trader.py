from stock_pricing import stockPrice
import trading_strategy as ts
import matplotlib.pyplot as plt


# Settings to run the trading strategies
ticker = "TSLA"     # Ticker symbol
period = "1y"       # Time period for fetching historical stock price data
forecast_days = 10  # Number of days to forecast into the future (used in regression prediction)
trading_strategy = "Moving Average"  # Trading strategy to use
plot = True

# Initialize the stock price object, fetch data for the specified period, and plot the stock price
company = stockPrice(ticker)
company.set_period(period)
company.get_stock_data()

manager = ts.tradeManager(company.ticker, company.stock_data)
manager.run_strategy(strategy_name=trading_strategy, plot=plot)

manager.backtest_strategy(plot=plot)

if plot:
    plt.show()
