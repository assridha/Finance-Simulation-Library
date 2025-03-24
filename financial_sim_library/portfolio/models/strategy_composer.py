from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ...option_simulator.strategies.base import OptionContract, StockPosition, StrategyPosition
from ...option_simulator.strategies import SimpleStrategy

class StrategyComposer(ABC):
    """Base class for strategy composers that generate option strategy positions."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def create_strategy(self, symbol: str, current_price: float, expiry_date: Any, 
                       fetcher: Any) -> SimpleStrategy:
        """
        Create a specific option strategy with the given parameters.
        
        Args:
            symbol: Stock ticker symbol
            current_price: Current price of the underlying
            expiry_date: Expiration date for the options
            fetcher: MarketDataFetcher instance to fetch market data
            
        Returns:
            SimpleStrategy object with the positions set up
        """
        pass 