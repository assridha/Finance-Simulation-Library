from typing import List
from .base import OptionStrategy, StrategyPosition, OptionContract

class PoorMansCoveredCall(OptionStrategy):
    """Poor Man's Covered Call: Long deep ITM call + Short OTM call."""
    
    def __init__(self, long_call: OptionContract, short_call: OptionContract):
        super().__init__("Poor Man's Covered Call")
        self.positions = [
            StrategyPosition(long_call, 1, long_call.premium),
            StrategyPosition(short_call, -1, short_call.premium)
        ]
    
    def calculate_payoff(self, stock_price: float) -> float:
        long_call_payoff = max(0, stock_price - self.positions[0].contract.strike_price)
        short_call_payoff = -max(0, stock_price - self.positions[1].contract.strike_price)
        return long_call_payoff + short_call_payoff
    
    def calculate_profit_loss(self, stock_price: float) -> float:
        return self.calculate_payoff(stock_price) - (self.positions[0].entry_price - self.positions[1].entry_price)
    
    def get_break_even_points(self) -> List[float]:
        return [self.positions[0].contract.strike_price + 
                (self.positions[0].entry_price - self.positions[1].entry_price)]
    
    def get_max_profit(self) -> float:
        return (self.positions[1].contract.strike_price - self.positions[0].contract.strike_price + 
                self.positions[1].entry_price - self.positions[0].entry_price)
    
    def get_max_loss(self) -> float:
        return self.positions[0].entry_price - self.positions[1].entry_price 