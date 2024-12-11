# Algorithmic Trading with Python

Repository that contains code to conduct some algorithmic trading, using 'yfinance' to import historical stock price data, and perform some simple trading strategies.

Linear Regression models from Scikit Learn are used to make forecasting predictions.

- Moving average strategy involves determining buy and sell positions based on the crossover of short-term and long-term moving averages.
- Hidden Markov strategy uses Hidden Markov Models (HMM) to determine a market state and make trading decisions based on the predicted state transitions
- Donchian Channel strategy involves using the highest high and the lowest low over a specific period to create upper and lower bands. A buy signal is generated when the price breaks above the upper band, and a sell signal is generated when the price falls below the lower band.

To use, open `trader.py`, and change the settings as desired.
- set the ticker to the company (e.g. "AAPL" for Apple)
- set the period to a time period of interest
- set the number of forecast days for the regression model to predict into the future

Also toggle the binary settings of whether to run certain features.
