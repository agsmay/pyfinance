import option_pricing_models as opt
from stock_pricing import stockPrice
import numpy as np
import matplotlib.pyplot as plt

stock_data = stockPrice('AAPL')
stock_data.set_period('1y')
stock_data.get_stock_data()
stock_data.plot_stock_price()

S = 100  # Current stock price
K = 105  # Strike price
tau = 1  # Time to maturity (1 year)
r = 0.05  # Risk-free rate (5%)
sigma = 0.2  # Volatility (20%)

# Black-Scholes model testing
BSM = opt.blackScholesModel(S, K, tau, r, sigma)

# Generate data
strike_prices = np.linspace(200, 1000, 100)  # Strike prices range
times_to_expiry = np.linspace(0.1,700, 100)  # Time to expiry range (years)
market_price = stock_data.get_current_price()
BSM.volatility_surface(strike_prices, times_to_expiry, market_price)


# Call volatility_smile for this specific slice
# BSM.volatility_smile(strike_prices, market_prices_for_tau, tau=tau, option_type="call")
# strike_prices = np.linspace(0.5, 1.5, 50)*S  # Range of strike prices
# times_to_expiry = np.linspace(0.01, 2, 50)  # Range of times to expiry

# # Plot the call price surface
# BSM.plot_price_surface(strike_prices, times_to_expiry, option_type="call")

# # Binomial model testing
# BOPM = opt.binomialTreeModel(100, 100, 365, 0.1, 0.2, 15000)
# print(BOPM.calculate_option_price('Call Option'))
# print(BOPM.calculate_option_price('Put Option'))

# # Monte Carlo simulation testing
# MC = opt.monteCarloModel(100, 100, 365, 0.1, 0.2, 10000)
# MC.simulate_prices()
# print(MC.calculate_option_price('Call Option'))
# print(MC.calculate_option_price('Put Option'))
# MC.plot_simulation_results(10)

plt.show()
