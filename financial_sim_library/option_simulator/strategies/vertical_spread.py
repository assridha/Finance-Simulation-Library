from typing import List
from .base import OptionStrategy, StrategyPosition, OptionContract

class VerticalSpread(OptionStrategy):
    """Vertical Spread strategy (can be call or put)."""
    
    def __init__(self, long_option: OptionContract, short_option: OptionContract):
        super().__init__("Vertical Spread")
        self.positions = [
            StrategyPosition(long_option, 1, long_option.premium),
            StrategyPosition(short_option, -1, short_option.premium)
        ]
    
    def calculate_payoff(self, stock_price: float) -> float:
        long_payoff = max(0, stock_price - self.positions[0].contract.strike_price) if self.positions[0].contract.option_type == 'call' else max(0, self.positions[0].contract.strike_price - stock_price)
        short_payoff = -max(0, stock_price - self.positions[1].contract.strike_price) if self.positions[1].contract.option_type == 'call' else -max(0, self.positions[1].contract.strike_price - stock_price)
        return long_payoff + short_payoff
    
    def calculate_profit_loss(self, stock_price: float) -> float:
        return self.calculate_payoff(stock_price) - (self.positions[0].entry_price - self.positions[1].entry_price)
    
    def get_break_even_points(self) -> List[float]:
        if self.positions[0].contract.option_type == 'call':
            return [self.positions[0].contract.strike_price + 
                   (self.positions[0].entry_price - self.positions[1].entry_price)]
        else:
            return [self.positions[0].contract.strike_price - 
                   (self.positions[0].entry_price - self.positions[1].entry_price)]
    
    def get_max_profit(self) -> float:
        if self.positions[0].contract.option_type == 'call':
            return (self.positions[1].contract.strike_price - self.positions[0].contract.strike_price + 
                   self.positions[1].entry_price - self.positions[0].entry_price)
        else:
            return (self.positions[0].contract.strike_price - self.positions[1].contract.strike_price + 
                   self.positions[1].entry_price - self.positions[0].entry_price)
    
    def get_max_loss(self) -> float:
        return self.positions[0].entry_price - self.positions[1].entry_price 