from enum import Enum
from abc import ABC, abstractmethod

class OPTION_TYPE(Enum):
    CALL_OPTION = 'Call Option'
    PUT_OPTION = 'Put Option'

class OptionPricingModel(ABC):
    """Abstract class defining interface for option pricing models."""

    def calculate_option_price(self, option_type):
        """Calculates call/put option price according to the specified parameter."""
        if option_type == OPTION_TYPE.CALL_OPTION.value:
            return self.calculate_call_price()
        elif option_type == OPTION_TYPE.PUT_OPTION.value:
            return self.calculate_put_price()
        else:
            return -1

    @abstractmethod
    def calculate_call_price(self):
        """Calculates option price for call option."""
        pass

    @abstractmethod
    def calculate_put_price(self):
        """Calculates option price for put option."""
        pass
