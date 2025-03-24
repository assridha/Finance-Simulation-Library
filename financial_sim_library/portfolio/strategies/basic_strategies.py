from ..models.strategy_composer import StrategyComposer
from ...option_simulator.strategies import SimpleStrategy

class SimpleCallComposer(StrategyComposer):
    """Composer for a simple long call option strategy."""
    
    def __init__(self):
        super().__init__("Simple Buy Call Option")
    
    def create_strategy(self, symbol, current_price, expiry_date, fetcher):
        """Create a simple long call option strategy."""
        # Get ATM option
        atm_options = fetcher.get_atm_options(symbol, expiry_date)
        call_contract = atm_options['call']
        
        # Define positions
        positions = [
            {'type': 'option', 'contract': call_contract, 'quantity': 1, 'name': 'Long Call'}
        ]
        
        # Create the strategy
        return SimpleStrategy(self.name, positions), positions
    
    def calculate_cost_basis(self, positions):
        """Calculate cost basis for simple long call option."""
        # For a long call, maximum loss is the premium paid
        net_premium = 0
        breakeven_points = []
        
        if positions and positions[0]['type'] == 'option':
            contract = positions[0]['contract']
            quantity = positions[0]['quantity']
            net_premium = contract.premium * quantity * 100
            
            # Breakeven is strike + premium
            if contract.option_type.lower() == 'call':
                breakeven_points = [contract.strike_price + contract.premium]
            
        return {
            'maximum_loss': abs(net_premium),
            'breakeven_points': breakeven_points,
            'net_premium': net_premium
        }


class CoveredCallComposer(StrategyComposer):
    """Composer for a covered call strategy (long stock + short call)."""
    
    def __init__(self):
        super().__init__("Covered Call Strategy")
    
    def create_strategy(self, symbol, current_price, expiry_date, fetcher):
        """Create a covered call strategy."""
        # Find OTM call with target delta around 0.3 (common for covered calls)
        otm_call = fetcher.find_option_by_delta(
            symbol,
            expiry_date,
            'call',
            0.3  # Target delta for OTM call
        )
        
        # Define positions
        positions = [
            {'type': 'stock', 'symbol': symbol, 'quantity': 100, 'entry_price': current_price},  # Long 100 shares
            {'type': 'option', 'contract': otm_call, 'quantity': -1, 'name': 'Short Call'}  # Short 1 call
        ]
        
        # Create the strategy
        return SimpleStrategy(self.name, positions), positions
    
    def calculate_cost_basis(self, positions):
        """Calculate cost basis for covered call strategy."""
        # Maximum loss is stock price - premium received (if stock goes to 0)
        net_premium = 0
        stock_cost = 0
        breakeven_points = []
        
        # Find stock and call positions
        stock_position = next((p for p in positions if p.get('type') == 'stock'), None)
        call_position = next((p for p in positions if p.get('type') != 'stock' and 
                              'contract' in p and p['contract'].option_type.lower() == 'call'), None)
        
        if stock_position and call_position:
            stock_entry_price = stock_position['entry_price']
            premium_received = call_position['contract'].premium * abs(call_position['quantity'])
            net_premium = -premium_received * 100  # Negative because credit received
            stock_cost = stock_entry_price * stock_position['quantity']
            
            # Maximum loss is stock cost minus premium received
            max_loss = stock_cost - abs(net_premium)
            
            # Breakeven is stock cost - premium received (per share)
            breakeven_points = [stock_entry_price - (premium_received * 100 / stock_position['quantity'])]
            
            return {
                'maximum_loss': max_loss,
                'breakeven_points': breakeven_points,
                'net_premium': net_premium
            }
            
        # Fallback to default calculation
        return super().calculate_cost_basis(positions)


class PoorMansCoveredCallComposer(StrategyComposer):
    """Composer for a poor man's covered call strategy (long deep ITM call + short OTM call)."""
    
    def __init__(self):
        super().__init__("Poor Man's Covered Call Strategy")
    
    def create_strategy(self, symbol, current_price, expiry_date, fetcher):
        """Create a poor man's covered call strategy."""
        # Get deep ITM call (high delta) and OTM call (low delta)
        itm_call = fetcher.find_option_by_delta(
            symbol,
            expiry_date,
            'call',
            0.8  # Deep ITM call delta
        )
        
        otm_call = fetcher.find_option_by_delta(
            symbol,
            expiry_date,
            'call',
            0.3  # OTM call delta
        )
        
        # Define positions
        positions = [
            {'type': 'option', 'contract': itm_call, 'quantity': 1, 'name': 'Long Option Contract'},  # Long deep ITM call
            {'type': 'option', 'contract': otm_call, 'quantity': -1, 'name': 'Short Option Contract'}  # Short OTM call
        ]
        
        # Create the strategy
        return SimpleStrategy(self.name, positions), positions
    
    def calculate_cost_basis(self, positions):
        """Calculate cost basis for poor man's covered call strategy."""
        # Maximum loss is the net debit paid for the spread
        net_premium = 0
        
        # Find long and short call positions
        long_call = next((p for p in positions if p.get('quantity', 0) > 0 and 'contract' in p), None)
        short_call = next((p for p in positions if p.get('quantity', 0) < 0 and 'contract' in p), None)
        
        if long_call and short_call:
            long_premium = long_call['contract'].premium * long_call['quantity'] * 100
            short_premium = short_call['contract'].premium * short_call['quantity'] * 100
            net_premium = long_premium + short_premium
            
            # Calculate breakeven (long strike + net debit per share)
            long_strike = long_call['contract'].strike_price
            net_debit = abs(net_premium) / 100
            breakeven_points = [long_strike + net_debit]
            
            return {
                'maximum_loss': abs(net_premium),
                'breakeven_points': breakeven_points,
                'net_premium': net_premium
            }
            
        # Fallback to default calculation
        return super().calculate_cost_basis(positions)


class VerticalSpreadComposer(StrategyComposer):
    """Composer for a vertical spread strategy (long call + short call at higher strike)."""
    
    def __init__(self):
        super().__init__("Vertical Spread Strategy")
    
    def create_strategy(self, symbol, current_price, expiry_date, fetcher):
        """Create a vertical spread strategy."""
        # Get 2 OTM call and OTM call for vertical spread
        otm_call1 = fetcher.find_option_by_delta(
            symbol,
            expiry_date,
            'call',
            0.2  # OTM call delta
        )
        
        otm_call2 = fetcher.find_option_by_delta(
            symbol,
            expiry_date,
            'call',
            0.1  # OTM call delta
        )
        
        # Define positions
        positions = [
            {'type': 'option', 'contract': otm_call1, 'quantity': 1, 'name': 'Long Option Contract'},  # Long ATM call
            {'type': 'option', 'contract': otm_call2, 'quantity': -1, 'name': 'Short Option Contract'}  # Short OTM call
        ]
        
        # Create the strategy
        return SimpleStrategy(self.name, positions), positions
    
    def calculate_cost_basis(self, positions):
        """Calculate cost basis for vertical spread strategy."""
        # Need to determine if this is a credit or debit spread and call or put vertical
        net_premium = 0
        
        # Sort options by strike price
        sorted_options = sorted([p for p in positions if 'contract' in p], 
                              key=lambda p: p['contract'].strike_price)
        
        if len(sorted_options) >= 2:
            # Calculate net premium
            for pos in positions:
                if pos.get('type') != 'stock' and 'contract' in pos:
                    contract = pos['contract']
                    quantity = pos['quantity']
                    position_cost = contract.premium * quantity * 100
                    net_premium += position_cost
            
            # Get strike prices
            low_strike = sorted_options[0]['contract'].strike_price
            high_strike = sorted_options[-1]['contract'].strike_price
            strike_width = high_strike - low_strike
            
            # Determine if credit or debit spread and calculate max loss
            max_loss = 0
            if net_premium > 0:  # Debit spread (paid premium)
                max_loss = abs(net_premium)
            else:  # Credit spread (received premium)
                max_loss = strike_width * 100 - abs(net_premium)
            
            # Calculate breakeven based on spread type
            first_contract = sorted_options[0]['contract']
            option_type = first_contract.option_type.lower()
            
            # Calculate breakeven points
            breakeven_points = []
            if option_type == 'call':
                if net_premium > 0:  # Bull call spread 
                    breakeven_points = [low_strike + (abs(net_premium) / 100)]
                else:  # Bear call spread
                    breakeven_points = [high_strike - (abs(net_premium) / 100)]
            else:  # Put
                if net_premium > 0:  # Bull put spread
                    breakeven_points = [high_strike - (abs(net_premium) / 100)]
                else:  # Bear put spread
                    breakeven_points = [low_strike + (abs(net_premium) / 100)]
            
            return {
                'maximum_loss': max_loss,
                'breakeven_points': breakeven_points,
                'net_premium': net_premium
            }
            
        # Fallback to default calculation
        return super().calculate_cost_basis(positions)


class ButterflySpreadComposer(StrategyComposer):
    """Composer for a butterfly spread strategy (using both calls and puts)."""
    
    def __init__(self):
        super().__init__("Butterfly Spread Strategy")
    
    def create_strategy(self, symbol, current_price, expiry_date, fetcher):
        """Create a proper butterfly spread strategy with both calls and puts."""
        # Get option chain
        chain = fetcher.get_option_chain(symbol, expiry_date)
        calls = chain[chain['option_type'] == 'call'].copy()  # Use copy to avoid SettingWithCopyWarning
        puts = chain[chain['option_type'] == 'put'].copy()  # Use copy to avoid SettingWithCopyWarning
        
        if calls.empty or puts.empty:
            raise ValueError(f"Call or put options not available for {symbol} at {expiry_date}")
        
        # Find ATM strike (closest to current price)
        calls['strike_diff'] = abs(calls['strike'] - current_price)
        puts['strike_diff'] = abs(puts['strike'] - current_price)
        
        atm_call_index = calls['strike_diff'].idxmin()
        atm_put_index = puts['strike_diff'].idxmin()
        
        atm_call_row = calls.loc[atm_call_index]
        atm_put_row = puts.loc[atm_put_index]
        
        atm_strike = atm_call_row['strike']  # Should be the same or very close to atm_put_row['strike']
        
        # Determine wing width based on available strikes
        wing_width = 10  # Target $10 wide wings
        
        # Find lower wing for put (OTM put)
        target_lower = atm_strike - wing_width
        puts['lower_diff'] = abs(puts['strike'] - target_lower)
        lower_put_index = puts['lower_diff'].idxmin()
        lower_put_row = puts.loc[lower_put_index]
        lower_strike = lower_put_row['strike']
        
        # Find upper wing for call (OTM call)
        target_upper = atm_strike + wing_width
        calls['upper_diff'] = abs(calls['strike'] - target_upper)
        upper_call_index = calls['upper_diff'].idxmin()
        upper_call_row = calls.loc[upper_call_index]
        upper_strike = upper_call_row['strike']
        
        # Show selected strikes
        print(f"Selected strikes for butterfly: {lower_strike}-{atm_strike}-{upper_strike}")
        
        # Create option contracts
        lower_put_contract = fetcher.create_option_contract(lower_put_row, symbol, expiry_date)
        atm_call_contract = fetcher.create_option_contract(atm_call_row, symbol, expiry_date)
        atm_put_contract = fetcher.create_option_contract(atm_put_row, symbol, expiry_date)
        upper_call_contract = fetcher.create_option_contract(upper_call_row, symbol, expiry_date)
        
        # Define positions for butterfly spread
        positions = [
            # Call wing (Buy OTM call, sell 1 ATM call)
            {'type': 'option', 'contract': upper_call_contract, 'quantity': 1, 'name': 'Upper Call Wing'},
            {'type': 'option', 'contract': atm_call_contract, 'quantity': -1, 'name': 'ATM Call'},
            
            # Put wing (Buy OTM put, sell 1 ATM put)
            {'type': 'option', 'contract': lower_put_contract, 'quantity': 1, 'name': 'Lower Put Wing'},
            {'type': 'option', 'contract': atm_put_contract, 'quantity': -1, 'name': 'ATM Put'},
        ]
        
        # Create the strategy
        return SimpleStrategy(self.name, positions), positions
    
    def calculate_cost_basis(self, positions):
        """Calculate cost basis for butterfly spread strategy."""
        # Butterfly spread: maximum loss is the net premium paid
        net_premium = 0
        
        # Calculate net premium for all positions
        for pos in positions:
            if pos.get('type') != 'stock' and 'contract' in pos:
                contract = pos['contract']
                quantity = pos['quantity']
                position_cost = contract.premium * quantity * 100
                net_premium += position_cost
        
        # Extract all option positions and sort by strike
        option_positions = [p for p in positions if 'contract' in p]
        sorted_options = sorted(option_positions, key=lambda p: p['contract'].strike_price)
        
        # Calculate breakeven points
        breakeven_points = []
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
        
        return {
            'maximum_loss': abs(net_premium),
            'breakeven_points': breakeven_points,
            'net_premium': net_premium
        } 