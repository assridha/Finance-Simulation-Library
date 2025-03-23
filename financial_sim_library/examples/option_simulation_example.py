import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ..option_simulator.data_fetcher import MarketDataFetcher
from ..option_simulator.strategies import SimpleStrategy
from ..option_simulator.simulator import MonteCarloOptionSimulator
from ..option_simulator.base import StockPosition, StrategyPosition
from ..utils.data_fetcher import find_closest_expiry_date, fetch_option_chain
import numpy as np

def print_option_contract_data(contract):
    """Print details of an option contract."""
    print("\n Option Contract:")
    print(f"Symbol: {contract.symbol}")
    print(f"Type: {contract.option_type.upper()}")
    print(f"Strike Price: ${contract.strike_price:.2f}")
    print(f"Premium: ${contract.premium:.2f}")
    print(f"Underlying Price: ${contract.underlying_price:.2f}")
    print(f"Expiration: {contract.expiration_date.strftime('%Y-%m-%d')}")

def print_strategy_positions(positions):
    """Print details of strategy positions."""
    print("\nStock Positions:")
    for pos in positions:
        if pos.get('type') == 'stock':
            print(f"Quantity: {pos['quantity']} shares")
    
    print("\nOption Positions:")
    for pos in positions:
        if pos.get('type') != 'stock':
            print(f"\nQuantity: {pos['quantity']}")
            print_option_contract_data(pos['contract'])

def plot_simulation_results(results, strategy_name):
    """Plot simulation results."""
    if results is None:
        return
    
    price_paths = results['price_paths']
    strategy_values = results['strategy_values']
    time_steps = results['time_steps']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Calculate dates for x-axis
    start_date = datetime.now()
    dates = [start_date + timedelta(days=int(t * 365)) for t in time_steps]
    
    # Plot stock price paths
    num_paths_to_plot = min(100, len(price_paths))
    for i in range(num_paths_to_plot):
        ax1.plot(dates, price_paths[i], alpha=0.1, color='blue')
    
    # Plot mean and percentiles for stock prices
    mean_price = np.mean(price_paths, axis=0)
    percentile_5_price = np.percentile(price_paths, 5, axis=0)
    percentile_95_price = np.percentile(price_paths, 95, axis=0)
    
    ax1.plot(dates, mean_price, 'r-', label='Mean Price')
    ax1.plot(dates, percentile_5_price, 'k--', label='5th Percentile')
    ax1.plot(dates, percentile_95_price, 'k--', label='95th Percentile')
    
    ax1.set_title(f'{strategy_name} - Stock Price Simulation')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot strategy P&L
    for i in range(num_paths_to_plot):
        ax2.plot(dates, strategy_values[i], alpha=0.1, color='blue')
    
    # Plot mean and percentiles for strategy values
    mean_value = np.mean(strategy_values, axis=0)
    percentile_5 = np.percentile(strategy_values, 5, axis=0)
    percentile_95 = np.percentile(strategy_values, 95, axis=0)
    
    ax2.plot(dates, mean_value, 'r-', label='Mean P&L')
    ax2.plot(dates, percentile_5, 'k--', label='5th Percentile')
    ax2.plot(dates, percentile_95, 'k--', label='95th Percentile')
    
    ax2.set_title(f'{strategy_name} - Strategy P&L')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Strategy P&L ($)')
    ax2.legend()
    ax2.grid(True)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

def get_target_expiry_date() -> datetime:
    """Get a target expiry date that's at least a month away."""
    current_date = datetime.now()
    target_date = current_date + timedelta(days=30)  # At least a month away
    return target_date.replace(year=2025, month=5, day=6)  # Example target date

def main():
    symbol = "AAPL"
    print(f"\nFetching market data for {symbol}...")
    
    fetcher = MarketDataFetcher()
    current_price = fetcher.get_stock_price(symbol)
    print(f"Current price: ${current_price:.2f}\n")
    
    # Get target expiry date
    target_expiry = get_target_expiry_date()
    print(f"Target expiry date: {target_expiry.strftime('%Y-%m-%d')}")
    
    # Get option chain to check available dates
    chain_data = fetch_option_chain(symbol)
    if not chain_data:
        print("No option chain data available.")
        return
    
    available_dates = list(chain_data.keys())
    target_expiry_str = target_expiry.strftime('%Y-%m-%d')
    actual_expiry_str = find_closest_expiry_date(target_expiry_str, available_dates)
    actual_expiry = datetime.strptime(actual_expiry_str, '%Y-%m-%d')
    print(f"Using closest available expiry date: {actual_expiry_str}\n")
    
    # Calculate number of days to expiry
    days_to_expiry = (actual_expiry - datetime.now()).days
    num_steps = min(252, days_to_expiry)  # Use fewer steps if expiry is closer
    
    # Simulate Simple Buy Call Option
    print("Simulating Simple Buy Call Option strategy...")
    try:
        atm_options = fetcher.get_atm_options(symbol, actual_expiry)
        call_contract = atm_options['call']
        
        print("\nOption Contract Details:")
        print_option_contract_data(call_contract)
        
        # Define positions for Simple Buy Call
        positions = [
            {'contract': call_contract, 'quantity': 1}
        ]
        
        print("\nStrategy Positions:")
        print_strategy_positions(positions)
        
        # Create strategy and simulator
        strategy = SimpleStrategy("Simple Buy Call", positions)
        simulator = MonteCarloOptionSimulator(
            strategy=strategy,
            price_model='gbm',
            volatility=fetcher.get_historical_volatility(symbol),
            risk_free_rate=fetcher.get_risk_free_rate()
        )
        
        # Run simulation
        results = simulator.run_simulation(num_paths=1000, num_steps=num_steps)
        plot_simulation_results(results, "Simple Buy Call Option")
    except Exception as e:
        print(f"Error simulating Simple Buy Call Option: {str(e)}")
    
    # Simulate Covered Call
    print("\nSimulating Covered Call strategy...")
    try:
        strategy_contracts = fetcher.get_option_strategy_contracts(
            symbol, 'covered_call', actual_expiry
        )
        
        print("\nOption Contract Details:")
        print_option_contract_data(strategy_contracts['call'])
        
        # Define positions for Covered Call
        positions = [
            {'type': 'stock', 'symbol': symbol, 'quantity': 100, 'entry_price': current_price},  # Long 100 shares
            {'contract': strategy_contracts['call'], 'quantity': -1}  # Short 1 call
        ]
        
        print("\nStrategy Positions:")
        print_strategy_positions(positions)
        
        # Create strategy and simulator
        strategy = SimpleStrategy("Covered Call", positions)
        simulator = MonteCarloOptionSimulator(
            strategy=strategy,
            price_model='gbm',
            volatility=fetcher.get_historical_volatility(symbol),
            risk_free_rate=fetcher.get_risk_free_rate()
        )
        
        # Run simulation
        results = simulator.run_simulation(num_paths=1000, num_steps=num_steps)
        plot_simulation_results(results, "Covered Call")
    except Exception as e:
        print(f"Error simulating Covered Call: {str(e)}")
    
    # Simulate Poor Man's Covered Call
    print("\nSimulating Poor Man's Covered Call strategy...")
    try:
        pmcc_data = fetcher.get_option_strategy_contracts(
            symbol, 'poor_mans_covered_call', actual_expiry
        )
        
        print("\nStrategy: Poor Man's Covered Call")
        print("\nStock Positions:")
        print("\nOption Positions:")
        print("\nLong Option Contract:")
        print_option_contract_data(pmcc_data['long_call'])
        print("\nShort Option Contract:")
        print_option_contract_data(pmcc_data['short_call'])
        
        # Define positions for PMCC
        positions = [
            {'contract': pmcc_data['long_call'], 'quantity': 1},  # Long deep ITM call
            {'contract': pmcc_data['short_call'], 'quantity': -1}  # Short OTM call
        ]
        
        # Create strategy and simulator
        strategy = SimpleStrategy("Poor Man's Covered Call", positions)
        simulator = MonteCarloOptionSimulator(
            strategy=strategy,
            price_model='gbm',
            volatility=fetcher.get_historical_volatility(symbol),
            risk_free_rate=fetcher.get_risk_free_rate()
        )
        
        # Run simulation
        results = simulator.run_simulation(num_paths=1000, num_steps=num_steps)
        plot_simulation_results(results, "Poor Man's Covered Call")
    except Exception as e:
        print(f"Error simulating Poor Man's Covered Call: {str(e)}")
    
    # Simulate Vertical Spread
    print("\nSimulating Vertical Spread strategy...")
    try:
        vertical_data = fetcher.get_option_strategy_contracts(
            symbol, 'vertical_spread', actual_expiry
        )
        
        print("\nStrategy: Vertical Spread")
        print("\nStock Positions:")
        print("\nOption Positions:")
        print("\nLong Option Contract:")
        print_option_contract_data(vertical_data['long_call'])
        print("\nShort Option Contract:")
        print_option_contract_data(vertical_data['short_call'])
        
        # Define positions for vertical spread
        positions = [
            {'contract': vertical_data['long_call'], 'quantity': 1},  # Long ATM call
            {'contract': vertical_data['short_call'], 'quantity': -1}  # Short OTM call
        ]
        
        # Create strategy and simulator
        strategy = SimpleStrategy("Vertical Spread", positions)
        simulator = MonteCarloOptionSimulator(
            strategy=strategy,
            price_model='gbm',
            volatility=fetcher.get_historical_volatility(symbol),
            risk_free_rate=fetcher.get_risk_free_rate()
        )
        
        # Run simulation
        results = simulator.run_simulation(num_paths=1000, num_steps=num_steps)
        plot_simulation_results(results, "Vertical Spread")
    except Exception as e:
        print(f"Error simulating Vertical Spread: {str(e)}")
    
    # Simulate Custom Butterfly Spread
    print("\nSimulating Custom Butterfly Spread strategy...")
    try:
        # Get option chain for butterfly spread
        chain = fetcher.get_option_chain(symbol, actual_expiry)
        calls = chain[chain['option_type'] == 'call']
        
        # Find strikes for butterfly wings
        atm_strike = calls[calls['strike'] >= current_price].iloc[0]['strike']
        wing_width = 10  # $10 wide wings
        
        lower_strike = atm_strike - wing_width
        upper_strike = atm_strike + wing_width
        
        # Get option contracts for butterfly spread
        lower_call = calls[calls['strike'] == lower_strike].iloc[0]
        atm_call = calls[calls['strike'] == atm_strike].iloc[0]
        upper_call = calls[calls['strike'] == upper_strike].iloc[0]
        
        # Create option contracts
        lower_contract = fetcher.create_option_contract(lower_call, symbol, actual_expiry)
        atm_contract = fetcher.create_option_contract(atm_call, symbol, actual_expiry)
        upper_contract = fetcher.create_option_contract(upper_call, symbol, actual_expiry)
        
        print("\nStrategy: Custom Butterfly Spread")
        print("\nStock Positions:")
        print("\nOption Positions:")
        print("\nLower Wing Call:")
        print_option_contract_data(lower_contract)
        print("\nBody Call:")
        print_option_contract_data(atm_contract)
        print("\nUpper Wing Call:")
        print_option_contract_data(upper_contract)
        
        # Define positions for butterfly spread
        positions = [
            {'contract': lower_contract, 'quantity': 1},  # Long lower wing
            {'contract': atm_contract, 'quantity': -2},   # Short body
            {'contract': upper_contract, 'quantity': 1}   # Long upper wing
        ]
        
        # Create strategy and simulator
        strategy = SimpleStrategy("Custom Butterfly Spread", positions)
        simulator = MonteCarloOptionSimulator(
            strategy=strategy,
            price_model='gbm',
            volatility=fetcher.get_historical_volatility(symbol),
            risk_free_rate=fetcher.get_risk_free_rate()
        )
        
        # Run simulation
        results = simulator.run_simulation(num_paths=1000, num_steps=num_steps)
        plot_simulation_results(results, "Custom Butterfly Spread")
    except Exception as e:
        print(f"Error simulating Custom Butterfly Spread: {str(e)}")

if __name__ == "__main__":
    main() 