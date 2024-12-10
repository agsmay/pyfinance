from stock_pricing import stockPrice
import trading_strategy as ts

class stockPortfolio:
    """
    Stock portfolio object to manage a list of stocks and a trading strategy.
    """
    def __init__(self, tickers, strategy):
        """
        Initialize the stock portfolio object with a list of tickers and a trading strategy.
        """

        self.tickers = tickers
        self.stocks = {ticker: stockPrice(ticker) for ticker in tickers}
        self.strategy = strategy
