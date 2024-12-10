import numpy as np
import pandas as pd
import pandas_ta as ta
from hmmlearn import hmm
import alpaca_trade_api as tradeapi

# Alpaca API setup
API_KEY = 'PKIE877S26EG7G4RWBSZ'
SECRET_KEY = '0UEFGidrBhwoY9OLZ6KrbkSSnbAKJCfnvaXT0tSK'
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Test the connection by fetching account information
account = api.get_account()
print(f"Account status: {account.status}")
print(f"Account balance: ${account.cash}")

# Fetch account balance and calculate 2% risk per trade
def get_risk_per_trade():
    account = api.get_account()
    account_balance = float(account.cash)
    risk_per_trade = account_balance * 0.02  # 2% risk
    return risk_per_trade, account_balance

# Fetch current price for a stock
def get_current_price(symbol):
    barset = api.get_barset(symbol, 'minute', 1)
    return barset[symbol][0].c  # Close price

# Fetch 5-minute historical data for AAPL
def get_historical_data(symbol, timeframe='5Min', limit=500):
    barset = api.get_barset(symbol, timeframe, limit=limit)
    return barset[symbol]

# Calculate position size, stop-loss, and take-profits (scaling out at 2:1 and 3:1 RR)
def calculate_trade_parameters(entry_price, risk_per_trade):
    position_size = risk_per_trade / entry_price
    stop_loss_price = entry_price - (risk_per_trade / position_size)
    take_profit_price_1 = entry_price + 2 * (entry_price - stop_loss_price)
    take_profit_price_2 = entry_price + 3 * (entry_price - stop_loss_price)
    return position_size, stop_loss_price, take_profit_price_1, take_profit_price_2

# Place the scaled-out trade
def place_scaled_trade(symbol, position_size, entry_price, stop_loss_price, take_profit_price_1, take_profit_price_2, is_long=True):
    side = 'buy' if is_long else 'sell'
    
    # First half of the position (2:1 RR)
    first_half_position = position_size / 2
    api.submit_order(
        symbol=symbol,
        qty=first_half_position,
        side=side,
        type='market',
        time_in_force='gtc',
        order_class='bracket',
        take_profit={'limit_price': take_profit_price_1},
        stop_loss={'stop_price': stop_loss_price}
    )

    # Second half of the position (3:1 RR)
    second_half_position = position_size / 2
    api.submit_order(
        symbol=symbol,
        qty=second_half_position,
        side=side,
        type='market',
        time_in_force='gtc',
        order_class='bracket',
        take_profit={'limit_price': take_profit_price_2},
        stop_loss={'stop_price': stop_loss_price}
    )

# Check if RSI signals a reversal and close position early
def check_rsi_and_exit(symbol, rsi_value, open_position, is_long):
    if open_position:
        if is_long and rsi_value > 70:
            print(f"RSI overbought. Closing long position for {symbol}.")
            api.close_position(symbol)
        elif not is_long and rsi_value < 30:
            print(f"RSI oversold. Closing short position for {symbol}.")
            api.close_position(symbol)

# Implement the trading strategy with HMM and RSI
def implement_trading_strategy(symbol='AAPL'):
    # Fetch historical data (5-minute bars)
    historical_data = get_historical_data(symbol, '5Min', 500)
    
    # Create a DataFrame from the historical data
    df = pd.DataFrame({
        'close': [bar.c for bar in historical_data],
        'high': [bar.h for bar in historical_data],
        'low': [bar.l for bar in historical_data],
        'open': [bar.o for bar in historical_data],
        'volume': [bar.v for bar in historical_data]
    })
    
    # Calculate price differences and RSI (shorter RSI period for lower time frames)
    df['price_diff'] = df['close'].diff().dropna()
    df['RSI'] = ta.rsi(df['close'], length=9).dropna()  # Shorter RSI period for faster trades
    
    # Drop NaN values
    df = df.dropna()

    # Prepare features for HMM: price differences and RSI
    features = np.column_stack([df['price_diff'].values, df['RSI'].values])

    # Define and train the HMM with 2 hidden states (bullish and bearish)
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)
    model.fit(features)

    # Predict hidden market states
    hidden_states = model.predict(features)

    # Fetch the latest price for AAPL
    entry_price = get_current_price(symbol)
    
    # Calculate risk per trade (2% of account balance)
    risk_per_trade, account_balance = get_risk_per_trade()

    # Calculate position size, stop loss, and take profits (2:1 and 3:1 RR)
    position_size, stop_loss_price, take_profit_price_1, take_profit_price_2 = calculate_trade_parameters(entry_price, risk_per_trade)

    # Get the current RSI
    current_rsi = df['RSI'].iloc[-1]

    # Determine trading action based on HMM states and RSI
    last_state = hidden_states[-1]
    
    # Long trade: Bullish state and RSI < 30 (oversold)
    if last_state == 1 and current_rsi < 30:
        print(f"Placing Buy order for {symbol} with stop-loss at {stop_loss_price} and scaling out at 2:1 and 3:1.")
        place_scaled_trade(symbol, position_size, entry_price, stop_loss_price, take_profit_price_1, take_profit_price_2, is_long=True)

    # Short trade: Bearish state and RSI > 70 (overbought)
    elif last_state == 0 and current_rsi > 70:
        print(f"Placing Sell order for {symbol} with stop-loss at {stop_loss_price} and scaling out at 2:1 and 3:1.")
        place_scaled_trade(symbol, position_size, entry_price, stop_loss_price, take_profit_price_1, take_profit_price_2, is_long=False)

    # Check if the RSI reaches the opposite extreme for closing the position early
    if last_state == 1:
        check_rsi_and_exit(symbol, current_rsi, open_position=True, is_long=True)
    elif last_state == 0:
        check_rsi_and_exit(symbol, current_rsi, open_position=True, is_long=False)

# Example usage of the strategy
implement_trading_strategy('AAPL')