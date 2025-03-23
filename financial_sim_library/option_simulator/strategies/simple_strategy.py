from typing import List
from ..base import OptionStrategy, OptionContract, StrategyPosition, StockPosition

class SimpleStrategy(OptionStrategy):
    """A simple option strategy that can hold multiple positions."""
    
    def __init__(self, name: str, positions: List[dict]):
        super().__init__(name)
        
        # Add positions
        for pos in positions:
            if pos.get('type') == 'stock':
                self.stock_positions.append(
                    StockPosition(
                        symbol=pos.get('symbol', 'STOCK'),
                        quantity=pos['quantity'],
                        entry_price=pos.get('entry_price', 0)
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
    
    def calculate_profit_loss(self, current_price: float) -> float:
        """Calculate profit/loss including initial premiums."""
        # Calculate total payoff
        payoff = self.calculate_payoff(current_price)
        
        # Calculate initial cost (premiums paid - premiums received)
        initial_cost = 0
        for pos in self.positions:
            if pos.get('type') == 'stock':
                # For stock positions, include the entry price
                initial_cost += pos.get('entry_price', 0) * pos.get('quantity', 0)
            else:
                # For option positions, include the premium
                contract = pos.get('contract')
                if contract:
                    initial_cost += contract.premium * pos.get('quantity', 0)
        
        # P&L is payoff minus initial cost
        return payoff - initial_cost
    
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