from stock_pricing import stockPrice
import trading_strategy as ts

class stockPortfolio:
    def __init__(self, tickers, strategy):
        self.tickers = tickers
        self.stocks = {ticker: stockPrice(ticker) for ticker in tickers}
        self.strategy = strategy

