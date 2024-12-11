from stock_pricing import stockPrice
import trading_strategy as ts
import matplotlib.pyplot as plt


# Settings to run the trading strategies
ticker = "AAPL"     # Ticker symbol
period = "5y"       # Time period for fetching historical stock price data
forecast_days = 10  # Number of days to forecast into the future

# Trading strategy settings
run_regression = False  # Run regression prediction
moving_average = False  # Run moving average trading strategy
hmm = False             # Run Hidden Markov Model trading strategy
donchian = True        # Run Donchian Channel trading strategy
plot = True            # Plot the results


# Initialize the stock price object, fetch data for the specified period, and plot the stock price
company = stockPrice(ticker)
company.set_period(period)
company.get_stock_data()

if run_regression:
    company.run_regression_prediction(forecast_days=forecast_days)

if moving_average:
    # Initialize and simulate the moving average trading strategy
    mov_average_strategy = ts.movingAverage(company.ticker, company.stock_data)
    mov_average_strategy.simulate_strategy(short_window=20, long_window=50)

    mov_average_strategy.backtest()
    mov_average_strategy.plot_backtest()

if hmm:
    # Initialize Hidden Markov Model strategy
    hmm_strategy = ts.hiddenMarkov(ticker="TSLA", stock_data=company.stock_data)

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

if donchian:
    # Initialize Donchian Channel strategy
    donchian_strategy = ts.donchianChannel(ticker="TSLA", stock_data=company.stock_data)

    # Simulate Donchian Channel strategy

    # Plot Donchian Channel strategy results
    donchian_strategy.simulate_strategy()
    donchian_strategy.plot_donchian_channel()

if plot:
    plt.show()
