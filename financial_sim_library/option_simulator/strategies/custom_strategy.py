from typing import List
from .base import OptionStrategy, StrategyPosition, StockPosition

class CustomStrategy(OptionStrategy):
    """Custom option strategy that can be built from multiple positions."""
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def add_position(self, position: StrategyPosition):
        """Add a position to the strategy."""
        self.positions.append(position)
    
    def add_stock_position(self, position: StockPosition):
        """Add a stock position to the strategy."""
        self.stock_positions.append(position)
    
    def calculate_payoff(self, stock_price: float) -> float:
        payoff = 0
        
        # Calculate stock position payoffs
        for stock_pos in self.stock_positions:
            payoff += stock_pos.quantity * (stock_price - stock_pos.entry_price)
        
        # Calculate option position payoffs
        for pos in self.positions:
            if pos.contract.option_type == 'call':
                payoff += pos.quantity * max(0, stock_price - pos.contract.strike_price)
            else:
                payoff += pos.quantity * max(0, pos.contract.strike_price - stock_price)
        
        return payoff
    
    def calculate_profit_loss(self, stock_price: float) -> float:
        return self.calculate_payoff(stock_price) - sum(pos.quantity * pos.entry_price for pos in self.positions)
    
    def get_break_even_points(self) -> List[float]:
        # This is a simplified version. For complex strategies, you might want to implement
        # a more sophisticated method to find break-even points.
        return [self.calculate_profit_loss(0)]
    
    def get_max_profit(self) -> float:
        # This is a simplified version. For complex strategies, you might want to implement
        # a more sophisticated method to find maximum profit.
        return float('inf')
    
    def get_max_loss(self) -> float:
        # This is a simplified version. For complex strategies, you might want to implement
        # a more sophisticated method to find maximum loss.
        return float('-inf') 