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