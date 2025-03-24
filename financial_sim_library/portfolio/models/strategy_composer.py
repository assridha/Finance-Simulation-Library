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
        
    def calculate_cost_basis(self, positions: List[Dict]) -> Dict[str, Any]:
        """
        Calculate the maximum potential loss (cost basis) for this strategy.
        
        Each strategy subclass should implement this method to calculate:
        - Maximum potential loss
        - Breakeven points
        - Other strategy-specific metrics
        
        Args:
            positions: List of position dictionaries from the strategy
            
        Returns:
            Dictionary with cost basis details including maximum loss and breakeven points
        """
        # Default implementation for generic strategies
        # Calculate net premium for all option positions
        net_premium = 0
        stock_cost = 0
        
        for pos in positions:
            if pos.get('type') != 'stock' and 'contract' in pos:
                contract = pos['contract']
                quantity = pos['quantity']
                position_cost = contract.premium * quantity * 100  # 100 shares per contract
                net_premium += position_cost
            elif pos.get('type') == 'stock':
                # Track stock position cost
                stock_cost += pos['entry_price'] * pos['quantity']
        
        # Calculate absolute value of negative cash flow (debits)
        debit_value = 0
        credit_value = 0
        
        for pos in positions:
            if pos.get('type') != 'stock' and 'contract' in pos:
                contract = pos['contract']
                quantity = pos['quantity']
                position_value = contract.premium * abs(quantity) * 100
                
                if quantity > 0:  # Long position (debit)
                    debit_value += position_value
                else:  # Short position (credit)
                    credit_value += position_value
            elif pos.get('type') == 'stock':
                # Stock positions
                if pos['quantity'] > 0:  # Long stock
                    debit_value += abs(pos['entry_price'] * pos['quantity'])
                else:  # Short stock
                    credit_value += abs(pos['entry_price'] * pos['quantity'])
        
        # Maximum loss is usually the net debit paid
        max_loss = max(0, debit_value - credit_value)
        
        # Prepare result dictionary
        return {
            'maximum_loss': max_loss,
            'breakeven_points': [],
            'net_premium': net_premium,
        } 