from typing import List
from .base import OptionStrategy, StrategyPosition, OptionContract

class NakedOption(OptionStrategy):
    """Naked option strategy (long or short call/put)."""
    
    def __init__(self, option: OptionContract, is_long: bool = True):
        super().__init__("Naked Option")
        self.positions = [StrategyPosition(option, 1 if is_long else -1, option.premium)]
    
    def calculate_payoff(self, stock_price: float) -> float:
        if self.positions[0].contract.option_type == 'call':
            return self.positions[0].quantity * max(0, stock_price - self.positions[0].contract.strike_price)
        else:
            return self.positions[0].quantity * max(0, self.positions[0].contract.strike_price - stock_price)
    
    def calculate_profit_loss(self, stock_price: float) -> float:
        return self.calculate_payoff(stock_price) - self.positions[0].quantity * self.positions[0].entry_price
    
    def get_break_even_points(self) -> List[float]:
        if self.positions[0].contract.option_type == 'call':
            return [self.positions[0].contract.strike_price + self.positions[0].entry_price]
        else:
            return [self.positions[0].contract.strike_price - self.positions[0].entry_price]
    
    def get_max_profit(self) -> float:
        return float('inf') if self.positions[0].quantity > 0 else self.positions[0].entry_price
    
    def get_max_loss(self) -> float:
        return self.positions[0].entry_price if self.positions[0].quantity > 0 else float('inf') 