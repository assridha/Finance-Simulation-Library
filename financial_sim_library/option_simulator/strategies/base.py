from dataclasses import dataclass
from typing import List
from datetime import datetime

@dataclass
class OptionContract:
    """Represents an option contract."""
    symbol: str
    strike_price: float
    expiration_date: datetime
    option_type: str  # 'call' or 'put'
    premium: float
    underlying_price: float

@dataclass
class StockPosition:
    """Represents a stock position."""
    symbol: str
    quantity: int
    entry_price: float

@dataclass
class StrategyPosition:
    """Represents a position in an option strategy."""
    contract: OptionContract
    quantity: int
    entry_price: float

class OptionStrategy:
    """Base class for option strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.positions: List[StrategyPosition] = []
        self.stock_positions: List[StockPosition] = []
    
    def calculate_payoff(self, stock_price: float) -> float:
        """Calculate the payoff at a given stock price."""
        raise NotImplementedError
    
    def calculate_profit_loss(self, stock_price: float) -> float:
        """Calculate profit/loss at a given stock price."""
        raise NotImplementedError
    
    def get_break_even_points(self) -> List[float]:
        """Get the break-even points for the strategy."""
        raise NotImplementedError
    
    def get_max_profit(self) -> float:
        """Get the maximum possible profit."""
        raise NotImplementedError
    
    def get_max_loss(self) -> float:
        """Get the maximum possible loss."""
        raise NotImplementedError 