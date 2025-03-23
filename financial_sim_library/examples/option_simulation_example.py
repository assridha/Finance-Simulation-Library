import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ..option_simulator.data_fetcher import MarketDataFetcher
from ..option_simulator.strategies import SimpleStrategy
from ..option_simulator.simulator import MonteCarloOptionSimulator
from ..option_simulator.strategies.base import StockPosition, StrategyPosition, OptionContract
from ..utils.data_fetcher import find_closest_expiry_date, fetch_option_chain
import numpy as np
import pandas as pd
from scipy.stats import norm
import argparse
import sys

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

def calculate_option_greeks(contract):
    """Calculate option Greeks for a given contract."""
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

def print_strategy_greeks(positions):
    """Calculate and print total Greeks for a strategy."""
    total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    # Calculate Greeks for each option position
    for pos in positions:
        if pos.get('type') != 'stock' and 'contract' in pos:
            contract = pos['contract']
            quantity = pos['quantity']
            greeks = calculate_option_greeks(contract)
            
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

def get_historical_volatility(symbol, days=30):
    """Calculate historical volatility for the specified number of days."""
    try:
        # Use the MarketDataFetcher's implementation
        fetcher = MarketDataFetcher()
        hist_vol = fetcher.get_historical_volatility(symbol)
        
        print(f"\nHistorical Volatility ({days} days): {hist_vol*100:.2f}%")
        return hist_vol
    except Exception as e:
        print(f"Error calculating historical volatility: {str(e)}")
        return None

def calculate_bid_ask_impact(positions):
    """Calculate the total bid-ask spread impact for a strategy.
    
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

def print_strategy_positions(positions, strategy_name):
    print(f"\nStrategy: {strategy_name}")
    
    print("\nStock Positions:")
    stock_positions = [p for p in positions if p['type'] == 'stock']
    for position in stock_positions:
        print(f"  {position['quantity']} shares of {position['symbol']} @ ${position['entry_price']:.2f}")
    
    print("\nOption Positions:")
    option_positions = [p for p in positions if p['type'] == 'option']
    for position in option_positions:
        contract = position['contract']
        print(f"\n{position.get('name', '')}:")
        print_option_contract_data(contract)
    
    # Calculate bid-ask impact
    bid_ask_impact = calculate_bid_ask_impact(positions)
    print(f"\nTotal Bid-Ask Impact: ${bid_ask_impact['total_cost']:.2f} ({bid_ask_impact['percentage_impact']:.2f}% of position value)")
    
    return stock_positions, option_positions

def plot_simulation_results(results, strategy_name, symbol, bid_ask_impact=None):
    """Plot simulation results including ticker symbol in titles."""
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
    
    # Add current price marker
    current_price = price_paths[0, 0]
    ax1.axhline(y=current_price, color='g', linestyle='--', alpha=0.8, label=f'Current Price (${current_price:.2f})')
    
    # Add expected move (±1 std) annotation
    last_date = dates[-1]
    std_dev = np.std(price_paths[:, -1])
    exp_move_pct = (std_dev / current_price) * 100
    ax1.annotate(f'Expected Move at Expiry: ±{exp_move_pct:.1f}%', 
                xy=(0.98, 0.05), xycoords='axes fraction', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    ax1.set_title(f'{symbol} - {strategy_name} - Stock Price Simulation')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot strategy values
    for i in range(num_paths_to_plot):
        ax2.plot(dates, strategy_values[i], alpha=0.1, color='blue')
    
    # Plot mean and percentiles for strategy values
    mean_value = np.mean(strategy_values, axis=0)
    percentile_5 = np.percentile(strategy_values, 5, axis=0)
    percentile_95 = np.percentile(strategy_values, 95, axis=0)
    
    ax2.plot(dates, mean_value, 'r-', label='Mean Value')
    ax2.plot(dates, percentile_5, 'k--', label='5th Percentile')
    ax2.plot(dates, percentile_95, 'k--', label='95th Percentile')
    
    # Add initial investment line
    initial_value = strategy_values[0, 0]
    ax2.axhline(y=initial_value, color='g', linestyle='--', alpha=0.8, 
                label=f'Initial Value (${initial_value:.2f})')
    
    # Calculate and display some key statistics
    final_values = strategy_values[:, -1]
    prob_profit = (final_values > initial_value).sum() / len(final_values) * 100
    max_value = np.max(final_values)
    min_value = np.min(final_values)
    exp_value = np.mean(final_values)
    
    # Add bid-ask spread impact to stats if provided
    if bid_ask_impact:
        stats_text = (
            f"Probability of Profit: {prob_profit:.1f}%\n"
            f"Expected Value: ${exp_value:.2f}\n"
            f"Max Value: ${max_value:.2f}\n"
            f"Min Value: ${min_value:.2f}\n"
            f"Return Ratio: {(max_value-initial_value)/(initial_value-min_value) if (initial_value-min_value) != 0 else float('inf'):.2f}\n"
            f"Bid-Ask Cost: ${bid_ask_impact['total_cost']:.2f} ({bid_ask_impact['percentage_impact']:.2f}%)"
        )
    else:
        stats_text = (
            f"Probability of Profit: {prob_profit:.1f}%\n"
            f"Expected Value: ${exp_value:.2f}\n"
            f"Max Value: ${max_value:.2f}\n"
            f"Min Value: ${min_value:.2f}\n"
            f"Return Ratio: {(max_value-initial_value)/(initial_value-min_value) if (initial_value-min_value) != 0 else float('inf'):.2f}"
        )
    
    ax2.annotate(stats_text, 
                xy=(0.02, 0.02), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    ax2.set_title(f'{symbol} - {strategy_name} - Strategy Value')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Strategy Value ($)')
    ax2.legend(loc='upper left')
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

def get_available_strategies():
    """Returns a dictionary of available strategy names and their descriptions."""
    return {
        'simple_call': 'Simple Buy Call Option',
        'covered_call': 'Covered Call Strategy',
        'pmcc': 'Poor Man\'s Covered Call Strategy',
        'vertical_spread': 'Vertical Spread Strategy',
        'butterfly': 'Custom Butterfly Spread Strategy',
        'all': 'Run all available strategies'
    }

def parse_arguments():
    """Parse command line arguments."""
    strategies = get_available_strategies()
    strategy_keys = list(strategies.keys())
    
    parser = argparse.ArgumentParser(description='Option Strategy Simulator')
    parser.add_argument('-s', '--symbol', type=str, default="AAPL",
                        help='Stock symbol to simulate (default: AAPL)')
    parser.add_argument('-n', '--num-paths', type=int, default=1000,
                        help='Number of paths to simulate (default: 1000)')
    parser.add_argument('-st', '--strategies', nargs='+', choices=strategy_keys, default=['all'],
                        help=f'Strategies to simulate. Available options: {", ".join(strategy_keys)}')
    
    args = parser.parse_args()
    
    # If 'all' is in the strategies list, run all strategies
    if 'all' in args.strategies:
        args.strategies = [s for s in strategy_keys if s != 'all']
    
    return args

def main():
    # Parse command line arguments
    args = parse_arguments()
    symbol = args.symbol
    num_paths = args.num_paths
    selected_strategies = args.strategies
    
    strategy_names = get_available_strategies()
    
    print(f"\nFetching market data for {symbol}...")
    
    fetcher = MarketDataFetcher()
    current_price = fetcher.get_stock_price(symbol)
    print(f"Current price: ${current_price:.2f}\n")
    
    # Get historical volatility for the last 30 days
    historical_vol = get_historical_volatility(symbol, days=30)
    
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
    print(f"Selected strategies: {', '.join(selected_strategy for selected_strategy in selected_strategies)}\n")
    
    # Calculate number of days to expiry
    days_to_expiry = (actual_expiry - datetime.now()).days
    num_steps = min(252, days_to_expiry)  # Use fewer steps if expiry is closer
    
    # Reusable simulation function to reduce code duplication
    def simulate_strategy(strategy_name, positions, volatility=None, risk_free_rate=None):
        """Run a Monte Carlo simulation for the given strategy."""
        print(f"\nSimulating {strategy_name} strategy...")
        try:
            # Print strategy Greeks
            print_strategy_greeks(positions)
            
            # Calculate bid-ask impact
            bid_ask_impact = calculate_bid_ask_impact(positions)
            
            # Create strategy and simulator
            strategy = SimpleStrategy(strategy_name, positions)
            
            # Use cached values if available
            vol = volatility or fetcher.get_historical_volatility(symbol)
            rate = risk_free_rate or fetcher.get_risk_free_rate()
            
            simulator = MonteCarloOptionSimulator(
                strategy=strategy,
                price_model='gbm',
                volatility=0.7*vol,
                risk_free_rate=rate,
                growth_rate=1.3
            )
            
            # Run simulation
            results = simulator.run_simulation(num_paths=num_paths, num_steps=num_steps)
            plot_simulation_results(results, strategy_name, symbol, bid_ask_impact)
            return results
        except Exception as e:
            print(f"Error simulating {strategy_name}: {str(e)}")
            return None
    
    # Get volatility and risk-free rate once for all simulations
    volatility = fetcher.get_historical_volatility(symbol)
    risk_free_rate = fetcher.get_risk_free_rate()
    
    # Simulate Simple Buy Call Option
    if 'simple_call' in selected_strategies:
        try:
            atm_options = fetcher.get_atm_options(symbol, actual_expiry)
            call_contract = atm_options['call']
            
            print("\nOption Contract Details:")
            print_option_contract_data(call_contract)
            
            # Define positions for Simple Buy Call
            positions = [
                {'type': 'option', 'contract': call_contract, 'quantity': 1, 'name': 'Long Call'}
            ]
            
            print("\nStrategy Positions:")
            stock_positions, option_positions = print_strategy_positions(positions, strategy_names['simple_call'])
            
            # Simulate
            simulate_strategy(strategy_names['simple_call'], positions, volatility, risk_free_rate)
        except Exception as e:
            print(f"Error preparing Simple Buy Call Option: {str(e)}")
    
    # Simulate Covered Call
    if 'covered_call' in selected_strategies:
        try:
            strategy_contracts = fetcher.get_option_strategy_contracts(
                symbol, 'covered_call', actual_expiry
            )
            
            print("\nOption Contract Details:")
            print_option_contract_data(strategy_contracts['call'])
            
            # Define positions for Covered Call
            positions = [
                {'type': 'stock', 'symbol': symbol, 'quantity': 100, 'entry_price': current_price},  # Long 100 shares
                {'type': 'option', 'contract': strategy_contracts['call'], 'quantity': -1, 'name': 'Short Call'}  # Short 1 call
            ]
            
            print("\nStrategy Positions:")
            stock_positions, option_positions = print_strategy_positions(positions, strategy_names['covered_call'])
            
            # Simulate
            simulate_strategy(strategy_names['covered_call'], positions, volatility, risk_free_rate)
        except Exception as e:
            print(f"Error preparing Covered Call: {str(e)}")
    
    # Simulate Poor Man's Covered Call
    if 'pmcc' in selected_strategies:
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
                {'type': 'option', 'contract': pmcc_data['long_call'], 'quantity': 1, 'name': 'Long Option Contract'},  # Long deep ITM call
                {'type': 'option', 'contract': pmcc_data['short_call'], 'quantity': -1, 'name': 'Short Option Contract'}  # Short OTM call
            ]
            
            # Simulate
            simulate_strategy(strategy_names['pmcc'], positions, volatility, risk_free_rate)
        except Exception as e:
            print(f"Error preparing Poor Man's Covered Call: {str(e)}")
    
    # Simulate Vertical Spread
    if 'vertical_spread' in selected_strategies:
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
                {'type': 'option', 'contract': vertical_data['long_call'], 'quantity': 1, 'name': 'Long Option Contract'},  # Long ATM call
                {'type': 'option', 'contract': vertical_data['short_call'], 'quantity': -1, 'name': 'Short Option Contract'}  # Short OTM call
            ]
            
            # Simulate
            simulate_strategy(strategy_names['vertical_spread'], positions, volatility, risk_free_rate)
        except Exception as e:
            print(f"Error preparing Vertical Spread: {str(e)}")
    
    # Simulate Custom Butterfly Spread
    if 'butterfly' in selected_strategies:
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
                {'type': 'option', 'contract': lower_contract, 'quantity': 1, 'name': 'Lower Wing Call'},  # Long lower wing
                {'type': 'option', 'contract': atm_contract, 'quantity': -2, 'name': 'Body Call'},   # Short body
                {'type': 'option', 'contract': upper_contract, 'quantity': 1, 'name': 'Upper Wing Call'}   # Long upper wing
            ]
            
            # Simulate
            simulate_strategy(strategy_names['butterfly'], positions, volatility, risk_free_rate)
        except Exception as e:
            print(f"Error preparing Custom Butterfly Spread: {str(e)}")

if __name__ == "__main__":
    main() 