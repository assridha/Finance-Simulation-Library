from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime

@dataclass
class OptionContract:
    """Represents an option contract with its key parameters."""
    symbol: str
    strike_price: float
    expiration_date: datetime
    option_type: str  # 'call' or 'put'
    premium: float
    underlying_price: float
    implied_volatility: float
    time_to_expiry: float  # in years
    risk_free_rate: float
    dividend_yield: float = 0.0

@dataclass
class StrategyPosition:
    """Represents a position in an option strategy."""
    contract: OptionContract
    quantity: int  # positive for long, negative for short
    entry_price: float

@dataclass
class StockPosition:
    """Represents a stock position."""
    symbol: str
    quantity: int  # positive for long, negative for short
    entry_price: float

class OptionStrategy(ABC):
    """Abstract base class for option strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.positions: List[StrategyPosition] = []
        self.stock_positions: List[StockPosition] = []
        
    @abstractmethod
    def calculate_payoff(self, stock_price: float) -> float:
        """Calculate the strategy payoff at a given stock price."""
        pass
    
    @abstractmethod
    def calculate_profit_loss(self, stock_price: float) -> float:
        """Calculate the profit/loss at a given stock price."""
        pass
    
    @abstractmethod
    def get_break_even_points(self) -> List[float]:
        """Calculate break-even points for the strategy."""
        pass
    
    @abstractmethod
    def get_max_profit(self) -> float:
        """Calculate maximum potential profit."""
        pass
    
    @abstractmethod
    def get_max_loss(self) -> float:
        """Calculate maximum potential loss."""
        pass

class OptionSimulator(ABC):
    """Abstract base class for option price simulation."""
    
    def __init__(self, strategy: OptionStrategy):
        self.strategy = strategy
        
    @abstractmethod
    def simulate_price_paths(self, 
                           num_paths: int, 
                           num_steps: int, 
                           time_to_expiry: float) -> np.ndarray:
        """Simulate multiple price paths for the underlying asset."""
        pass
    
    @abstractmethod
    def calculate_option_prices(self, 
                              price_paths: np.ndarray, 
                              time_steps: np.ndarray) -> np.ndarray:
        """Calculate option prices along each price path."""
        pass
    
    @abstractmethod
    def calculate_strategy_values(self, 
                                price_paths: np.ndarray, 
                                option_prices: np.ndarray) -> np.ndarray:
        """Calculate the total strategy value along each price path."""
        pass
    
    def run_simulation(self, 
                      num_paths: int = 1000, 
                      num_steps: int = 100) -> Dict:
        """Run a complete simulation of the strategy."""
        # Get the maximum time to expiry from all options
        max_time = max(pos.contract.time_to_expiry for pos in self.strategy.positions)
        
        # Simulate price paths
        price_paths = self.simulate_price_paths(num_paths, num_steps, max_time)
        
        # Create time steps array
        time_steps = np.linspace(0, max_time, num_steps)
        
        # Calculate option prices
        option_prices = self.calculate_option_prices(price_paths, time_steps)
        
        # Calculate strategy values
        strategy_values = self.calculate_strategy_values(price_paths, option_prices)
        
        return {
            'price_paths': price_paths,
            'option_prices': option_prices,
            'strategy_values': strategy_values,
            'time_steps': time_steps
        } 