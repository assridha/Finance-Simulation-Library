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

# Import the new portfolio module components
from ..portfolio import (
    StrategyAnalyzer, 
    SimpleCallComposer,
    CoveredCallComposer,
    PoorMansCoveredCallComposer,
    VerticalSpreadComposer,
    ButterflySpreadComposer
)

def print_option_contract_data(contract):
    """Print detailed information of an option contract."""
    StrategyAnalyzer.print_option_contract_data(contract)

def calculate_option_greeks(contract):
    """Calculate option Greeks for a given contract."""
    return StrategyAnalyzer.calculate_option_greeks(contract)

def print_strategy_greeks(positions):
    """Calculate and print total Greeks for a strategy."""
    return StrategyAnalyzer.print_strategy_greeks(positions)

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
    """Calculate the total bid-ask spread impact for a strategy."""
    # Use the StrategyAnalyzer's implementation
    return StrategyAnalyzer.calculate_bid_ask_impact(positions)

def print_strategy_positions(positions, strategy_name):
    """Print detailed information about strategy positions."""
    return StrategyAnalyzer.print_strategy_positions(positions, strategy_name)

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
        'butterfly': 'Butterfly Spread Strategy',
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
    parser.add_argument('-g', '--growth-rate', type=float, default=None,
                        help='Custom annual growth rate for simulations (default: risk-free rate)')
    parser.add_argument('-v', '--volatility-override', type=float, default=None,
                        help='Custom annual volatility for simulations (default: historical volatility)')
    parser.add_argument('-vm', '--volatility-multiplier', type=float, default=1.0,
                        help='Multiplier to apply to historical volatility (default: 1.0)')
    
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
    custom_growth_rate = args.growth_rate
    custom_volatility = args.volatility_override
    volatility_multiplier = args.volatility_multiplier
    
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
            
            # Get base volatility and risk-free rate
            vol = volatility or fetcher.get_historical_volatility(symbol)
            rate = risk_free_rate or fetcher.get_risk_free_rate()
            
            # Apply custom volatility if provided
            if custom_volatility is not None:
                vol = custom_volatility
                print(f"Using custom volatility: {vol*100:.2f}%")
            else:
                # Apply volatility multiplier if not using custom volatility
                vol = vol * volatility_multiplier
                if volatility_multiplier != 1.0:
                    print(f"Applied volatility multiplier: {volatility_multiplier}x → {vol*100:.2f}%")
            
            # Set growth rate
            growth = custom_growth_rate if custom_growth_rate is not None else rate
            if custom_growth_rate is not None:
                print(f"Using custom growth rate: {growth*100:.2f}%")
            
            simulator = MonteCarloOptionSimulator(
                strategy=strategy,
                price_model='gbm',
                volatility=vol,  # Use the actual volatility
                risk_free_rate=rate,
                growth_rate=growth  # Set growth_rate equal to risk-free rate (risk-neutral assumption)
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
    
    # Create strategy composers
    strategy_composers = {
        'simple_call': SimpleCallComposer(),
        'covered_call': CoveredCallComposer(),
        'pmcc': PoorMansCoveredCallComposer(),
        'vertical_spread': VerticalSpreadComposer(),
        'butterfly': ButterflySpreadComposer()
    }
    
    # Simulate each selected strategy
    for strategy_key in selected_strategies:
        try:
            print(f"\nPreparing {strategy_names[strategy_key]}...")
            
            # Get the appropriate composer
            composer = strategy_composers[strategy_key]
            
            # Create the strategy
            strategy, positions = composer.create_strategy(symbol, current_price, actual_expiry, fetcher)
            
            # Print details
            print_strategy_positions(positions, strategy_names[strategy_key])
            
            # Simulate
            simulate_strategy(strategy_names[strategy_key], positions, volatility, risk_free_rate)
        except Exception as e:
            print(f"Error preparing {strategy_names[strategy_key]}: {str(e)}")

if __name__ == "__main__":
    main() 