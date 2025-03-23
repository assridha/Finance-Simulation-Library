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

class SimpleStrategy(OptionStrategy):
    """A simple option strategy that can hold multiple positions."""
    
    def __init__(self, name: str, positions: List[dict]):
        super().__init__(name)
        
        # Add positions
        for pos in positions:
            if pos.get('type') == 'stock':
                self.stock_positions.append(
                    StockPosition(
                        symbol=pos['contract'].symbol if pos['contract'] else 'STOCK',
                        quantity=pos['quantity'],
                        entry_price=pos['contract'].underlying_price if pos['contract'] else 0
                    )
                )
            else:
                self.positions.append(
                    StrategyPosition(
                        contract=pos['contract'],
                        quantity=pos['quantity'],
                        entry_price=pos['contract'].premium
                    )
                )
    
    def calculate_payoff(self, stock_price: float) -> float:
        """Calculate the strategy payoff at a given stock price."""
        total_payoff = 0.0
        
        # Calculate stock position payoffs
        for pos in self.stock_positions:
            total_payoff += pos.quantity * (stock_price - pos.entry_price)
        
        # Calculate option position payoffs
        for pos in self.positions:
            if pos.contract.option_type == 'call':
                payoff = max(0, stock_price - pos.contract.strike_price)
            else:  # put
                payoff = max(0, pos.contract.strike_price - stock_price)
            total_payoff += pos.quantity * payoff
        
        return total_payoff
    
    def calculate_profit_loss(self, stock_price: float) -> float:
        """Calculate the profit/loss at a given stock price."""
        payoff = self.calculate_payoff(stock_price)
        
        # Subtract initial premiums paid/received
        for pos in self.positions:
            total_premium = pos.quantity * pos.entry_price
            payoff -= total_premium  # Subtract premium (negative for short positions)
        
        return payoff
    
    def get_break_even_points(self) -> List[float]:
        """Calculate break-even points for the strategy."""
        # This is a simplified implementation
        # In reality, we would need to solve for points where profit/loss = 0
        # For now, we'll return the strike prices as approximate break-even points
        break_even_points = []
        for pos in self.positions:
            break_even_points.append(pos.contract.strike_price)
        return sorted(list(set(break_even_points)))
    
    def get_max_profit(self) -> float:
        """Calculate maximum potential profit."""
        # This is a simplified implementation
        # In reality, we would need to analyze the payoff function
        return float('inf')  # Unlimited profit potential for most strategies
    
    def get_max_loss(self) -> float:
        """Calculate maximum potential loss."""
        # This is a simplified implementation
        # In reality, we would need to analyze the payoff function
        total_loss = 0.0
        for pos in self.positions:
            if pos.quantity > 0:  # Long position
                total_loss -= pos.quantity * pos.entry_price
            else:  # Short position
                if pos.contract.option_type == 'call':
                    total_loss = float('-inf')  # Unlimited loss potential
                    break
                else:
                    total_loss += abs(pos.quantity) * (
                        pos.contract.strike_price - pos.entry_price
                    )
        return total_loss 