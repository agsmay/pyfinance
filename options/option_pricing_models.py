import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from option_base import OptionPricingModel

class blackScholesModel(OptionPricingModel):
    """ 
    Class implementing calculation for European option price using Black-Scholes Formula.

    Call/Put option price is calculated with following assumptions:
    - European option can be exercised only on maturity date.
    - Underlying stock does not pay divident during option's lifetime.  
    - The risk free rate and volatility are constant.
    - Efficient Market Hypothesis - market movements cannot be predicted.
    - Lognormal distribution of underlying returns.
    """

    def __init__(self, S, K, tau, r, sigma):
        """
        Initialise the Black-Scholes model with the following parameters:
            S: Current price of the underlying asset
            K: Strike price of the option
            tau: Time to maturity of the option
            r: Risk-free interest rate
            sigma: Volatility of the underlying asset
        """
        
        self.S = S
        self.K = K
        self.tau = tau
        self.r = r
        self.sigma = sigma

    def calculate_call_price(self):
        """
        Calculate the price of a call option using the Black-Scholes formula
        """

        # cumulative function of standard normal distribution (risk-adjusted probability that the option will be exercised)     
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.tau) / (self.sigma * np.sqrt(self.tau))
        
        # cumulative function of standard normal distribution (probability of receiving the stock at expiration of the option)
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.tau) / (self.sigma * np.sqrt(self.tau))
        
        return (self.S * norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.tau) * norm.cdf(d2, 0.0, 1.0))
    
    def calculate_put_price(self):
        """
        Calculate the price of a put option using the Black-Scholes formula
        """

        # cumulative function of standard normal distribution (risk-adjusted probability that the option will be exercised)     
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.tau) / (self.sigma * np.sqrt(self.tau))
        
        # cumulative function of standard normal distribution (probability of receiving the stock at expiration of the option)
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.tau) / (self.sigma * np.sqrt(self.tau))
        
        return (self.K * np.exp(-self.r * self.tau) * norm.cdf(-d2, 0.0, 1.0) - self.S * norm.cdf(-d1, 0.0, 1.0))

    def calculate_implied_volatility(self, market_price, option_type="call", tol=1e-5, max_iter=100):
        """
        Calculate the implied volatility using the Newton-Raphson method.
        
        Parameters:
        - market_price: Observed market price of the option
        - option_type: "call" or "put"
        - tol: Tolerance for stopping criteria
        - max_iter: Maximum number of iterations
        """

        sigma = 0.2  # Initial guess for volatility
        for _ in range(max_iter):
            self.sigma = sigma
            if option_type == "call":
                price = self.calculate_call_price()
            elif option_type == "put":
                price = self.calculate_put_price()
            else:
                raise ValueError("Invalid option type. Use 'call' or 'put'.")

            # Vega is the derivative of the price with respect to volatility
            d1 = (np.log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.tau) / (sigma * np.sqrt(self.tau))
            vega = self.S * norm.pdf(d1) * np.sqrt(self.tau)

            # Newton-Raphson step
            price_diff = price - market_price
            if abs(price_diff) < tol:
                return sigma
            sigma -= price_diff / vega
        raise RuntimeError("Implied volatility did not converge")

    def plot_price_surface(self, strike_prices, times_to_expiry, option_type="call"):
        """
        Plot the surface of option prices as a function of strike price and time to expiry.
        
        Parameters:
        - strike_prices: array-like, range of strike prices
        - times_to_expiry: array-like, range of times to expiry
        - option_type: str, "call" or "put" option
        """
        K_grid, tau_grid = np.meshgrid(strike_prices, times_to_expiry)

        # Calculate option prices
        if option_type.lower() == "call":
            prices = np.array([self.calculate_call_price() for _ in range(len(K_grid.ravel()))])
        elif option_type.lower() == "put":
            prices = np.array([self.calculate_put_price() for _ in range(len(K_grid.ravel()))])
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        prices = prices.reshape(K_grid.shape)

        # Plot the surface
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_grid, tau_grid, prices, cmap='viridis', edgecolor='k', alpha=0.8)

        # Add labels and title
        ax.set_title(f'{option_type.capitalize()} Option Price Surface', fontsize=16)
        ax.set_xlabel('Strike Price ($K$)', fontsize=12)
        ax.set_ylabel('Time to Expiry ($\\tau$)', fontsize=12)
        ax.set_zlabel('Option Price', fontsize=12)

    def volatility_surface(self, strike_prices, times_to_expiry, market_price, option_type="call"):
        """
        Plots the implied volatility surface.
        :param strike_prices: Array of strike prices
        :param times_to_expiry: Array of times to expiry
        :param market_prices: 2D array of market prices for the options
        :param option_type: "call" or "put"
        """
        # Calculate implied volatilities
        implied_vols = np.zeros((len(times_to_expiry), len(strike_prices)))
        for i, tau in enumerate(times_to_expiry):
            for j, K in enumerate(strike_prices):
                try:
                    implied_vols[i, j] = self.calculate_implied_volatility(
                        market_price, option_type
                    )
                except RuntimeError as e:
                    implied_vols[i, j] = np.nan  # Handle non-convergence
                    print(f"Implied volatility failed to converge for K={K}, tau={tau}: {e}")

        # Plot volatility surface
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        K_grid, tau_grid = np.meshgrid(strike_prices, times_to_expiry)
        surface = ax.plot_surface(K_grid, tau_grid, implied_vols, cmap='viridis', edgecolor='k', alpha=0.8)

        # Add color bar
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label="Implied Volatility")

        # Labels and title
        ax.set_title("Implied Volatility Surface", fontsize=16)
        ax.set_xlabel("Strike Price ($K$)", fontsize=12)
        ax.set_ylabel("Time to Expiry ($\\tau$)", fontsize=12)
        ax.set_zlabel("Implied Volatility ($\\sigma_{implied}$)", fontsize=12)

    def volatility_smile(self, strike_prices, market_prices, tau=1, option_type="call"):
        """
        Plots the volatility smile for a fixed time to expiry.
        :param strike_prices: Array of strike prices
        :param market_prices: Array of market prices (one for each strike price)
        :param tau: Time to expiry (fixed)
        :param option_type: "call" or "put"
        """
        implied_vol_smile = []
        for K, market_price in zip(strike_prices, market_prices):
            try:
                vol = self.calculate_implied_volatility(market_price, option_type)
                implied_vol_smile.append(vol)
            except RuntimeError as e:
                implied_vol_smile.append(np.nan)
                print(f"Implied volatility failed to converge for K={K}, tau={tau}: {e}")

        # Plot the smile
        plt.figure(figsize=(10, 6))
        plt.plot(strike_prices, implied_vol_smile, label=f"Volatility Smile (tau={tau})", marker='o')
        plt.xlabel("Strike Price ($K$)", fontsize=12)
        plt.ylabel("Implied Volatility ($\\sigma_{implied}$)", fontsize=12)
        plt.title("Volatility Smile", fontsize=16)
        plt.legend()
        plt.grid(True)

    def plot_greeks(self):
        """
        Plot the Greeks (Delta, Gamma, Vega, Theta, Rho) for the option.
        """
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.tau) / (self.sigma * np.sqrt(self.tau))
        d2 = d1 - self.sigma * np.sqrt(self.tau)

        delta = norm.cdf(d1)  # Call option Delta
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.tau))
        vega = self.S * norm.pdf(d1) * np.sqrt(self.tau)
        theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.tau)) - self.r * self.K * np.exp(-self.r * self.tau) * norm.cdf(d2))
        rho = self.K * self.tau * np.exp(-self.r * self.tau) * norm.cdf(d2)

        greeks = {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

        # Plot Greeks
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        ax = ax.ravel()
        for i, (key, value) in enumerate(greeks.items()):
            ax[i].bar([0], [value], color="blue", alpha=0.7)
            ax[i].set_title(key)
            ax[i].set_ylim(0, max(greeks.values()) * 1.2)
            ax[i].set_xticks([])
        plt.tight_layout()
    
class binomialTreeModel(OptionPricingModel):
    """ 
    Class implementing calculation for European option price using BOPM (Binomial Option Pricing Model).
    It caclulates option prices in discrete time (lattice based), in specified number of time points between date of valuation and exercise date.
    This pricing model has three steps:
    - Price tree generation
    - Calculation of option value at each final node 
    - Sequential calculation of the option value at each preceding node
    """

    def __init__(self, S, K, tau, r, sigma, n):
        """
        Initialise the Binomial Tree model with the following parameters:
            S: Current price of the underlying asset
            K: Strike price of the option
            tau: Time to maturity of the option
            r: Risk-free interest rate
            sigma: Volatility of the underlying asset
            n: Number of time steps in the binomial tree
        """
        
        self.S = S
        self.K = K
        self.tau = tau
        self.r = r
        self.sigma = sigma
        self.n = n

    def calculate_call_price(self):
        """
        Calculate the price of a call option using the Binomial Tree model
        """

        dt = self.tau / self.n
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1.0 / u

        # Initialize the stock price tree
        V = np.zeros(self.n + 1)

        # Underlying asset price at time T
        S_T = np.array([(self.S * u**j * d**(self.n - j)) for j in range(self.n+1)])

        a = np.exp(self.r * dt)     # risk free compound return
        p = (a - d) / (u - d)       # risk neutral up probability
        q = 1 - p                   # risk neutral down probability

        V[:] = np.maximum(S_T - self.K, 0)  # call option payoff at maturity

        # Overiding option price
        for i in range(self.n-1, -1, -1):
            V[:-1] = np.exp(-self.r * dt) * (p * V[1:] + q * V[:-1]) # risk neutral pricing

        return V[0]
    
    def calculate_put_price(self):
        """Calculates price for put option according to the Binomial formula."""  
        # Delta t, up and down factors
        dt = self.tau / self.n                             
        u = np.exp(self.sigma * np.sqrt(dt))                 
        d = 1.0 / u                                    

        # Price vector initialization
        V = np.zeros(self.n + 1)                       

        # Underlying asset prices at different time points
        S_T = np.array( [(self.S * u**j * d**(self.n - j)) for j in range(self.n + 1)])

        a = np.exp(self.r * dt)      # risk free compounded return
        p = (a - d) / (u - d)        # risk neutral up probability
        q = 1.0 - p                  # risk neutral down probability   

        V[:] = np.maximum(self.K - S_T, 0.0)
    
        # Overriding option price 
        for i in range(self.n - 1, -1, -1):
            V[:-1] = np.exp(-self.r * dt) * (p * V[1:] + q * V[:-1]) 

        return V[0]
    
class monteCarloModel(OptionPricingModel):
    """ 
    Class implementing calculation for European option price using Monte Carlo Simulation.
    We simulate underlying asset price on expiry date using random stochastic process - Brownian motion.
    For the simulation generated prices at maturity, we calculate and sum up their payoffs, average them and discount the final value.
    That value represents option price
    """

    def __init__(self, S, K, tau, r, sigma, N):
        """
        Initialise the Monte Carlo model with the following parameters:
            S: Current price of the underlying asset
            K: Strike price of the option
            tau: Time to maturity of the option
            r: Risk-free interest rate
            sigma: Volatility of the underlying asset
            N: Number of simulations
        """
        
        self.S_0 = S
        self.K = K
        self.T = tau / 365
        self.r = r
        self.sigma = sigma
        self.N = N

        self.n = tau
        self.dt = self.T / self.n

    def simulate_prices(self):
        """
        Simulating price movement of underlying prices using Brownian random process.
        Saving random results.
        """
        np.random.seed(20)
        self.simulation_results = None

        # Initializing price movements for simulation: rows as time index and columns as different random price movements.
        S = np.zeros((self.n, self.N))        
        # Starting value for all price movements is the current spot price
        S[0] = self.S_0

        for t in range(1, self.n):
            # Random values to simulate Brownian motion (Gaussian distibution)
            Z = np.random.standard_normal(self.N)
            # Updating prices for next point in time 
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt + (self.sigma * np.sqrt(self.dt) * Z))

        self.simulation_results_S = S

    def calculate_call_price(self): 
        """
        Call option price calculation. Calculating payoffs for simulated prices at expiry date, summing up, averiging them and discounting.   
        Call option payoff (it's exercised only if the price at expiry date is higher than a strike price): max(S_t - K, 0)
        """
        if self.simulation_results_S is None:
            return -1
        return np.exp(-self.r * self.T) * 1 / self.N * np.sum(np.maximum(self.simulation_results_S[-1] - self.K, 0))
    
    def calculate_put_price(self): 
        """
        Put option price calculation. Calculating payoffs for simulated prices at expiry date, summing up, averiging them and discounting.   
        Put option payoff (it's exercised only if the price at expiry date is lower than a strike price): max(K - S_t, 0)
        """
        if self.simulation_results_S is None:
            return -1
        return np.exp(-self.r * self.T) * 1 / self.N * np.sum(np.maximum(self.K - self.simulation_results_S[-1], 0))

    def plot_simulation_results(self, num_of_movements):
        """Plots specified number of simulated price movements."""
        plt.figure(figsize=(12,8))
        plt.plot(self.simulation_results_S[:,0:num_of_movements])
        plt.axhline(self.K, c='k', xmin=0, xmax=self.n, label='Strike Price')
        plt.xlim([0, self.n])
        plt.ylabel('Simulated price movements')
        plt.xlabel('Days in future')
        plt.title(f'First {num_of_movements}/{self.N} Random Price Movements')
        plt.legend(loc='best')
