from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime
from scipy.stats import norm

class StrategyAnalyzer:
    """
    Utility class for analyzing and displaying information about option strategies.
    """
    
    @staticmethod
    def print_option_contract_data(contract):
        """Print detailed information of an option contract."""
        print("\n Option Contract:")
        print(f"Symbol: {contract.symbol}")
        print(f"Type: {contract.option_type.upper()}")
        print(f"Strike Price: ${contract.strike_price:.2f}")
        print(f"Premium: ${contract.premium:.2f}")
        print(f"Underlying Price: ${contract.underlying_price:.2f}")
        print(f"Implied Volatility: {contract.implied_volatility*100:.2f}%")
        print(f"Time to Expiry: {contract.time_to_expiry*365:.1f} days")
        print(f"Risk-Free Rate: {contract.risk_free_rate*100:.2f}%")
        print(f"Expiration: {contract.expiration_date.strftime('%Y-%m-%d')}")
        # Add bid-ask spread information
        print(f"Bid: ${contract.bid:.2f}")
        print(f"Ask: ${contract.ask:.2f}")
        print(f"Spread: ${contract.spread:.2f} ({contract.spread_percent:.2f}%)")
    
    @staticmethod
    def calculate_option_greeks(contract) -> Dict[str, float]:
        """
        Calculate option Greeks for a given contract.
        
        Args:
            contract: Option contract object
            
        Returns:
            Dictionary with calculated Greeks
        """
        S = contract.underlying_price
        K = contract.strike_price
        T = contract.time_to_expiry
        r = contract.risk_free_rate
        sigma = contract.implied_volatility
        q = contract.dividend_yield
        
        # Black-Scholes d1 and d2
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Call or put selection
        if contract.option_type.lower() == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
            theta = -((S * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))) * norm.pdf(d1) - \
                    r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # Put
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            theta = -((S * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))) * norm.pdf(d1) + \
                    r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        # Common Greeks for both call and put
        gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1) / 100  # Divided by 100 to get sensitivity per 1%
        
        # Each contract controls 100 shares
        return {
            'delta': delta * 100,  # Delta per contract
            'gamma': gamma * 100,  # Gamma per contract
            'theta': (theta / 365) * 100,  # Daily theta per contract
            'vega': vega * 100,  # Vega per contract
        }
    
    @staticmethod
    def calculate_bid_ask_impact(positions: List[Dict]) -> Dict[str, float]:
        """
        Calculate the total bid-ask spread impact for a strategy.
        
        Args:
            positions: List of position dictionaries containing option contracts
            
        Returns:
            Dictionary with total bid-ask cost and percentage impact
        """
        total_bid_ask_cost = 0.0
        total_position_value = 0.0
        
        for pos in positions:
            if pos.get('type') != 'stock' and 'contract' in pos:
                contract = pos['contract']
                quantity = abs(pos['quantity'])  # Absolute value since we care about total cost
                
                # Calculate half-spread cost (assumes execution at midpoint)
                half_spread = contract.spread / 2
                position_spread_cost = half_spread * quantity * 100  # 100 shares per contract
                
                # Add to total costs
                total_bid_ask_cost += position_spread_cost
                
                # Add position value for percentage calculation
                position_value = contract.premium * quantity * 100
                total_position_value += position_value
        
        # Calculate percentage impact
        if total_position_value > 0:
            percentage_impact = (total_bid_ask_cost / total_position_value) * 100
        else:
            percentage_impact = 0.0
        
        return {
            'total_cost': total_bid_ask_cost,
            'percentage_impact': percentage_impact
        }
    
    @staticmethod
    def calculate_cost_basis(positions: List[Dict], strategy_name: str) -> Dict[str, float]:
        """
        Calculate the maximum potential loss (cost basis) for a trade based on the options strategy.
        
        Args:
            positions: List of position dictionaries
            strategy_name: Name of the strategy for specific calculation logic
            
        Returns:
            Dictionary with cost basis details including maximum loss and breakeven points
        """
        # Initialize variables
        max_loss = 0
        breakeven_points = []
        cost_basis_details = {}
        net_premium = 0
        stock_cost = 0
        max_width = 0
        
        # Calculate net premium for all option positions
        for pos in positions:
            if pos.get('type') != 'stock' and 'contract' in pos:
                contract = pos['contract']
                quantity = pos['quantity']
                position_cost = contract.premium * quantity * 100  # 100 shares per contract
                net_premium += position_cost
            elif pos.get('type') == 'stock':
                # Track stock position cost
                stock_cost += pos['entry_price'] * pos['quantity']
        
        # Identify strategy type by name and calculate maximum loss
        if "Simple Buy Call" in strategy_name or "Simple Buy Put" in strategy_name:
            # Long option: maximum loss is the premium paid
            max_loss = abs(net_premium)
            # Breakeven for call is strike + premium
            # Breakeven for put is strike - premium
            if positions and 'contract' in positions[0]:
                contract = positions[0]['contract']
                premium_per_share = contract.premium
                if contract.option_type.lower() == 'call':
                    breakeven_points = [contract.strike_price + premium_per_share]
                else:
                    breakeven_points = [contract.strike_price - premium_per_share]
        
        elif "Covered Call" in strategy_name:
            # Covered call: maximum loss is stock price - premium received (if stock goes to 0)
            # Find stock and call positions
            stock_position = next((p for p in positions if p.get('type') == 'stock'), None)
            call_position = next((p for p in positions if p.get('type') != 'stock' and 
                                  'contract' in p and p['contract'].option_type.lower() == 'call'), None)
            
            if stock_position and call_position:
                stock_entry_price = stock_position['entry_price']
                premium_received = call_position['contract'].premium * abs(call_position['quantity'])
                max_loss = stock_entry_price * stock_position['quantity'] - (premium_received * 100)
                # Breakeven is stock cost - premium received
                breakeven_points = [stock_entry_price - (premium_received * 100 / stock_position['quantity'])]
            else:
                max_loss = abs(stock_cost) - abs(net_premium)
        
        elif "Poor Man's Covered Call" in strategy_name:
            # PMCC: maximum loss is the net debit paid for the spread
            max_loss = abs(net_premium)
            # Find long and short call for breakeven calculation
            long_call = next((p for p in positions if p.get('quantity', 0) > 0 and 'contract' in p), None)
            short_call = next((p for p in positions if p.get('quantity', 0) < 0 and 'contract' in p), None)
            
            if long_call and short_call:
                long_strike = long_call['contract'].strike_price
                short_strike = short_call['contract'].strike_price
                net_debit = long_call['contract'].premium * 100 - (short_call['contract'].premium * 100)
                # Breakeven is long strike + net debit
                breakeven_points = [long_strike + (net_debit / 100)]
        
        elif "Vertical Spread" in strategy_name:
            # Vertical spread: maximum loss is width of strikes - net premium received (for credit spread)
            # or net premium paid (for debit spread)
            sorted_options = sorted([p for p in positions if 'contract' in p], 
                                  key=lambda p: p['contract'].strike_price)
            
            if len(sorted_options) >= 2:
                low_strike = sorted_options[0]['contract'].strike_price
                high_strike = sorted_options[-1]['contract'].strike_price
                strike_width = high_strike - low_strike
                
                # Determine if credit or debit spread
                if net_premium > 0:  # Debit spread (paid premium)
                    max_loss = abs(net_premium)
                else:  # Credit spread (received premium)
                    max_loss = strike_width * 100 - abs(net_premium)
                
                # Breakeven calculation
                first_contract = sorted_options[0]['contract']
                if first_contract.option_type.lower() == 'call':
                    # Call vertical
                    if net_premium > 0:  # Bull call spread
                        breakeven_points = [low_strike + (abs(net_premium) / 100)]
                    else:  # Bear call spread
                        breakeven_points = [high_strike - (abs(net_premium) / 100)]
                else:
                    # Put vertical
                    if net_premium > 0:  # Bull put spread
                        breakeven_points = [high_strike - (abs(net_premium) / 100)]
                    else:  # Bear put spread
                        breakeven_points = [low_strike + (abs(net_premium) / 100)]
        
        elif "Butterfly" in strategy_name:
            # Butterfly spread: maximum loss is the net premium paid
            max_loss = abs(net_premium)
            
            # Extract all option positions and sort by strike
            option_positions = [p for p in positions if 'contract' in p]
            sorted_options = sorted(option_positions, key=lambda p: p['contract'].strike_price)
            
            if len(sorted_options) >= 3:
                # Get unique strike prices
                strikes = sorted(set(p['contract'].strike_price for p in option_positions))
                if len(strikes) >= 3:
                    # For butterfly, breakevens are at body ± net_premium_per_contract
                    middle_strike = strikes[len(strikes)//2]
                    wing_width = strikes[-1] - middle_strike  # Assuming symmetrical
                    
                    # Calculate net premium per wing width
                    net_premium_per_spread = abs(net_premium) / 100
                    
                    # Breakeven points
                    breakeven_points = [
                        middle_strike - wing_width + net_premium_per_spread,
                        middle_strike + wing_width - net_premium_per_spread
                    ]
        
        else:
            # Generic calculation for other strategies
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
        cost_basis_details = {
            'maximum_loss': max_loss,
            'breakeven_points': breakeven_points,
            'net_premium': net_premium,
        }
        
        return cost_basis_details
    
    @staticmethod
    def print_strategy_greeks(positions) -> Dict[str, float]:
        """
        Calculate and print total Greeks for a strategy.
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Dictionary with total strategy Greeks
        """
        total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        # Calculate Greeks for each option position
        for pos in positions:
            if pos.get('type') != 'stock' and 'contract' in pos:
                contract = pos['contract']
                quantity = pos['quantity']
                greeks = StrategyAnalyzer.calculate_option_greeks(contract)
                
                # Add weighted Greeks to the totals
                for greek, value in greeks.items():
                    total_greeks[greek] += value * quantity
            
            # Add delta for stock positions (delta=1)
            elif pos.get('type') == 'stock':
                total_greeks['delta'] += pos['quantity']
        
        # Print total strategy Greeks
        print("\nTotal Strategy Greeks:")
        print(f"Delta: {total_greeks['delta']:.2f}")
        print(f"Gamma: {total_greeks['gamma']:.5f}")
        print(f"Theta: ${total_greeks['theta']:.2f} per day")
        print(f"Vega: ${total_greeks['vega']:.2f} per 1% change in IV")
        
        return total_greeks
    
    @staticmethod
    def print_strategy_positions(positions, strategy_name) -> Tuple[List, List]:
        """
        Print detailed information about strategy positions.
        
        Args:
            positions: List of position dictionaries
            strategy_name: Name of the strategy
            
        Returns:
            Tuple of stock positions and option positions
        """
        print(f"\nStrategy: {strategy_name}")
        
        print("\nStock Positions:")
        stock_positions = [p for p in positions if p['type'] == 'stock']
        for position in stock_positions:
            quantity = position['quantity']
            position_type = "LONG" if quantity > 0 else "SHORT"
            print(f"  {position_type} {abs(quantity)} shares of {position['symbol']} @ ${position['entry_price']:.2f}")
        
        print("\nOption Positions:")
        option_positions = [p for p in positions if p['type'] == 'option']
        for position in option_positions:
            contract = position['contract']
            quantity = position['quantity']
            position_type = "BOUGHT" if quantity > 0 else "SOLD"
            contracts_text = "contract" if abs(quantity) == 1 else "contracts"
            
            print(f"\n{position.get('name', '')}:")
            print(f"  {position_type} {abs(quantity)} {contract.option_type.upper()} {contracts_text} @ ${contract.premium:.2f}")
            StrategyAnalyzer.print_option_contract_data(contract)
        
        # Calculate bid-ask impact using the analyzer's method
        bid_ask_impact = StrategyAnalyzer.calculate_bid_ask_impact(positions)
        print(f"\nTotal Bid-Ask Impact: ${bid_ask_impact['total_cost']:.2f} ({bid_ask_impact['percentage_impact']:.2f}% of position value)")
        
        # Calculate cost basis (maximum potential loss)
        cost_basis = StrategyAnalyzer.calculate_cost_basis(positions, strategy_name)
        print(f"\nCost Basis (Maximum Potential Loss): ${cost_basis['maximum_loss']:.2f}")
        
        # Print breakeven points if available
        if cost_basis['breakeven_points']:
            breakeven_points_str = ', '.join(f"${point:.2f}" for point in cost_basis['breakeven_points'])
            print(f"Breakeven Point(s): {breakeven_points_str}")
        
        return stock_positions, option_positions 