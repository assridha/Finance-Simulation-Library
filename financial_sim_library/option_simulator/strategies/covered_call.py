from typing import List
from .base import OptionStrategy, StrategyPosition, StockPosition, OptionContract

class CoveredCall(OptionStrategy):
    """Covered Call strategy: Long stock + Short call option."""
    
    def __init__(self, stock_position: StockPosition, call_option: OptionContract):
        super().__init__("Covered Call")
        self.stock_positions.append(stock_position)
        self.positions.append(StrategyPosition(call_option, -1, call_option.premium))
        
    def calculate_payoff(self, stock_price: float) -> float:
        stock_payoff = self.stock_positions[0].quantity * (stock_price - self.stock_positions[0].entry_price)
        option_payoff = -max(0, stock_price - self.positions[0].contract.strike_price)
        return stock_payoff + option_payoff
    
    def calculate_profit_loss(self, stock_price: float) -> float:
        return self.calculate_payoff(stock_price) - self.positions[0].entry_price
    
    def get_break_even_points(self) -> List[float]:
        return [self.stock_positions[0].entry_price - self.positions[0].entry_price]
    
    def get_max_profit(self) -> float:
        return (self.positions[0].contract.strike_price - self.stock_positions[0].entry_price + 
                self.positions[0].entry_price)
    
    def get_max_loss(self) -> float:
        return self.stock_positions[0].entry_price - self.positions[0].entry_price 