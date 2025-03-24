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

def print_strategy_positions(positions, strategy_name, strategy_composer=None):
    """Print detailed information about strategy positions."""
    return StrategyAnalyzer.print_strategy_positions(positions, strategy_name, strategy_composer)

def calculate_max_pnl_percentage(strategy_values, initial_value, cost_basis):
    """
    Calculate the maximum PnL% for each simulated path.
    
    Args:
        strategy_values: Array of strategy value paths from simulation
        initial_value: Initial value of the strategy
        cost_basis: Maximum potential loss (cost basis) of the strategy
        
    Returns:
        Array of max PnL% values for each path
    """
    max_values = np.max(strategy_values, axis=1)
    max_pnl_pct = (max_values - initial_value) / cost_basis * 100
    return max_pnl_pct

def plot_max_pnl_quantiles(max_pnl_pct, strategy_name, symbol):
    """Plot the max PnL% values against their quantiles."""
    # Create a new figure
    fig = plt.figure(figsize=(10, 6))
    
    # Sort the max_pnl_pct values
    sorted_pnl = np.sort(max_pnl_pct)
    
    # Calculate quantiles (0 to 1)
    quantiles = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl)
    
    # Calculate key statistics
    mean_pnl = np.mean(max_pnl_pct)
    median_pnl = np.median(max_pnl_pct)
    std_dev = np.std(max_pnl_pct)
    
    # Calculate percentiles
    p10 = np.percentile(max_pnl_pct, 10)
    p25 = np.percentile(max_pnl_pct, 25)
    p75 = np.percentile(max_pnl_pct, 75)
    p90 = np.percentile(max_pnl_pct, 90)
    p95 = np.percentile(max_pnl_pct, 95)
    p99 = np.percentile(max_pnl_pct, 99)
    
    # Plot quantile curve
    plt.plot(quantiles, sorted_pnl, 'b-', linewidth=2)
    
    # Add horizontal lines for mean and median
    plt.axhline(y=mean_pnl, color='r', linestyle='-', label=f'Mean: {mean_pnl:.2f}%')
    plt.axhline(y=median_pnl, color='g', linestyle='--', label=f'Median: {median_pnl:.2f}%')
    
    # Add vertical lines for key percentiles
    plt.axvline(x=0.1, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=0.25, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=0.75, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=0.9, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=0.95, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=0.99, color='gray', linestyle=':', alpha=0.5)
    
    # Add annotations for percentiles
    plt.annotate(f'10%: {p10:.2f}%', xy=(0.1, p10), xytext=(0.1, p10 + 10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), 
                horizontalalignment='center', verticalalignment='bottom')
    
    plt.annotate(f'25%: {p25:.2f}%', xy=(0.25, p25), xytext=(0.25, p25 + 10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), 
                horizontalalignment='center', verticalalignment='bottom')
    
    plt.annotate(f'75%: {p75:.2f}%', xy=(0.75, p75), xytext=(0.75, p75 + 10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), 
                horizontalalignment='center', verticalalignment='bottom')
    
    plt.annotate(f'90%: {p90:.2f}%', xy=(0.9, p90), xytext=(0.9, p90 + 10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), 
                horizontalalignment='center', verticalalignment='bottom')
    
    plt.annotate(f'95%: {p95:.2f}%', xy=(0.95, p95), xytext=(0.95, p95 + 10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), 
                horizontalalignment='center', verticalalignment='bottom')
    
    plt.annotate(f'99%: {p99:.2f}%', xy=(0.99, p99), xytext=(0.99, p99 + 10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), 
                horizontalalignment='center', verticalalignment='bottom')
    
    # Add statistics text box
    stats_text = (
        f"Mean: {mean_pnl:.2f}%\n"
        f"Median: {median_pnl:.2f}%\n"
        f"Std Dev: {std_dev:.2f}%\n"
        f"95th Percentile: {p95:.2f}%\n"
        f"99th Percentile: {p99:.2f}%"
    )
    
    plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Set title and labels
    plt.title(f'{symbol} - {strategy_name} - Max PnL% Distribution')
    plt.xlabel('Quantile')
    plt.ylabel('Max PnL%')
    plt.grid(True)
    plt.legend()
    
    # Return the figure instead of showing it
    return fig

def plot_simulation_results(results, strategy_name, symbol, bid_ask_impact=None, cost_basis=None):
    """Plot simulation results including ticker symbol in titles."""
    if results is None:
        return None
    
    price_paths = results['price_paths']
    strategy_values = results['strategy_values']
    time_steps = results['time_steps']
    
    # Create figure with two subplots
    main_fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
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
    
    # Create the quantile plot figure if cost basis is provided
    pnl_fig = None
    if cost_basis is not None and cost_basis > 0:
        # Calculate max PnL% for each path
        max_pnl_pct = calculate_max_pnl_percentage(strategy_values, initial_value, cost_basis)
        
        # Calculate key statistics
        mean_max_pnl = np.mean(max_pnl_pct)
        p95_max_pnl = np.percentile(max_pnl_pct, 95)
        
        # Add to stats text
        stats_text += f"\nAvg Max PnL%: {mean_max_pnl:.2f}%\n95th %tile Max PnL%: {p95_max_pnl:.2f}%"
        
        # Create quantile plot but don't show it yet
        pnl_fig = plot_max_pnl_quantiles(max_pnl_pct, strategy_name, symbol)
    
    ax2.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    ax2.set_title(f'{symbol} - {strategy_name} - Strategy Value')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Strategy Value ($)')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    main_fig.tight_layout()
    
    # Show all figures at once
    plt.show()
    
    # Return both figures for reference
    return main_fig, pnl_fig

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
    def simulate_strategy(strategy_name, positions, volatility=None, risk_free_rate=None, cost_basis=None):
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
            
            # Plot results with cost basis
            fig = plot_simulation_results(results, strategy_name, symbol, bid_ask_impact, cost_basis)
            return results, fig
        except Exception as e:
            print(f"Error simulating {strategy_name}: {str(e)}")
            return None, None
    
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
            
            # Pass the composer to print_strategy_positions to use its cost basis calculation
            stock_positions, option_positions = print_strategy_positions(positions, strategy_names[strategy_key], composer)
            
            # Get cost basis for Max PnL% calculation
            cost_basis_details = composer.calculate_cost_basis(positions)
            cost_basis = cost_basis_details['maximum_loss']
            
            # Simulate the strategy with Monte Carlo
            results, fig = simulate_strategy(
                strategy_names[strategy_key],
                positions, 
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                cost_basis=cost_basis
            )
        except Exception as e:
            print(f"Error preparing {strategy_names[strategy_key]}: {str(e)}")

if __name__ == "__main__":
    main() 