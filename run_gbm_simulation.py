#!/usr/bin/env python3
"""
Stock Price Simulator using Geometric Brownian Motion

This script simulates stock price paths using geometric Brownian motion
and visualizes the results.
"""

import argparse
import matplotlib.pyplot as plt
import yfinance as yf
from option_simulator import (
    simulate_geometric_brownian_motion,
    plot_price_simulations,
    calculate_historical_volatility,
    get_risk_free_rate
)
from datetime import datetime, timedelta
import logging
import numpy as np
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_simulation(ticker, days_to_simulate, num_simulations, benchmark=None, save_path=None):
    """
    Run a stock price simulation using geometric Brownian motion
    
    Args:
        ticker (str): Ticker symbol
        days_to_simulate (int): Number of days to simulate
        num_simulations (int): Number of simulation paths to generate
        benchmark (str, optional): Benchmark ticker to compare against (e.g., 'SPY')
        save_path (str, optional): Path to save the plot
    """
    # Get current stock price
    logger.info(f"Fetching current price for {ticker}")
    stock = yf.Ticker(ticker)
    current_price = stock.history(period="1d")['Close'].iloc[-1]
    logger.info(f"Current price of {ticker}: ${current_price:.2f}")
    
    # Calculate historical volatility
    historical_volatility = calculate_historical_volatility(ticker)
    logger.info(f"Historical volatility (30-day): {historical_volatility:.2%}")
    
    # Get current risk-free rate
    risk_free_rate = get_risk_free_rate()
    logger.info(f"Current risk-free rate: {risk_free_rate:.2%}")
    
    # Run the GBM simulation
    logger.info(f"Running simulation with {num_simulations} paths for {days_to_simulate} days")
    simulation_results = simulate_geometric_brownian_motion(
        ticker=ticker,
        current_price=current_price,
        days_to_simulate=days_to_simulate,
        num_simulations=num_simulations,
        historical_volatility=historical_volatility,
        risk_free_rate=risk_free_rate
    )
    
    # Extract future date for display
    future_date = (datetime.now().date() + timedelta(days=days_to_simulate)).strftime('%Y-%m-%d')
    logger.info(f"Simulation end date: {future_date}")
    
    # Calculate summary statistics at the end date
    final_prices = simulation_results['price_paths'][:, -1]
    mean_final = final_prices.mean()
    median_final = np.median(final_prices)
    min_final = final_prices.min()
    max_final = final_prices.max()
    
    logger.info(f"Simulated price statistics at {future_date}:")
    logger.info(f"  Mean: ${mean_final:.2f}")
    logger.info(f"  Median: ${median_final:.2f}")
    logger.info(f"  Min: ${min_final:.2f}")
    logger.info(f"  Max: ${max_final:.2f}")
    
    # Calculate percentiles for price targets
    percentiles = [10, 25, 50, 75, 90]
    price_targets = np.percentile(final_prices, percentiles)
    
    print(f"\nPrice Targets for {ticker} on {future_date}:")
    print("-" * 50)
    print(f"{'Percentile':^10} | {'Price Target':^15} | {'Change %':^10}")
    print("-" * 50)
    for i, percentile in enumerate(percentiles):
        price = price_targets[i]
        change_pct = (price / current_price - 1) * 100
        print(f"{percentile:^10}% | ${price:^13.2f} | {change_pct:^+10.2f}%")
    print("-" * 50)
    
    # Calculate benchmark simulation if requested
    benchmark_results = None
    if benchmark:
        logger.info(f"Simulating benchmark {benchmark} for comparison")
        benchmark_stock = yf.Ticker(benchmark)
        benchmark_price = benchmark_stock.history(period="1d")['Close'].iloc[-1]
        logger.info(f"Current price of {benchmark}: ${benchmark_price:.2f}")
        
        benchmark_vol = calculate_historical_volatility(benchmark)
        logger.info(f"{benchmark} historical volatility (30-day): {benchmark_vol:.2%}")
        
        # Calculate correlation between the stock and benchmark
        stock_hist = stock.history(period="1y")['Close'].pct_change().dropna()
        bench_hist = benchmark_stock.history(period="1y")['Close'].pct_change().dropna()
        
        # Align dates and calculate correlation
        common_dates = stock_hist.index.intersection(bench_hist.index)
        if len(common_dates) > 0:
            correlation = stock_hist.loc[common_dates].corr(bench_hist.loc[common_dates])
            logger.info(f"Correlation between {ticker} and {benchmark}: {correlation:.4f}")
        else:
            correlation = 0
            logger.warning(f"Could not calculate correlation between {ticker} and {benchmark}. Using 0.")
        
        # Run benchmark simulation
        benchmark_results = simulate_geometric_brownian_motion(
            ticker=benchmark,
            current_price=benchmark_price,
            days_to_simulate=days_to_simulate,
            num_simulations=num_simulations,
            historical_volatility=benchmark_vol,
            risk_free_rate=risk_free_rate
        )
        
        # Calculate relative performance statistics
        benchmark_final_prices = benchmark_results['price_paths'][:, -1]
        benchmark_mean_return = (benchmark_final_prices.mean() / benchmark_price - 1) * 100
        stock_mean_return = (mean_final / current_price - 1) * 100
        
        print(f"\nBenchmark Comparison ({ticker} vs {benchmark}):")
        print(f"Expected {ticker} return: {stock_mean_return:+.2f}%")
        print(f"Expected {benchmark} return: {benchmark_mean_return:+.2f}%")
        print(f"Alpha (excess return): {stock_mean_return - benchmark_mean_return:+.2f}%")
        
        # Probability of outperforming the benchmark
        relative_returns = (final_prices / current_price) - (benchmark_final_prices / benchmark_price)
        prob_outperform = (relative_returns > 0).mean()
        print(f"Probability of {ticker} outperforming {benchmark}: {prob_outperform:.2%}")
    
    # Plot the simulation results
    fig = plot_price_simulations(simulation_results, num_paths_to_plot=min(10, num_simulations), 
                                save_path=save_path, benchmark_results=benchmark_results)
    
    # Calculate probability of price being above/below certain thresholds
    print(f"\nProbability Analysis for {ticker} on {future_date}:")
    print(f"Probability of price above current (${current_price:.2f}): {(final_prices > current_price).mean():.2%}")
    print(f"Probability of price 10% higher (>${current_price * 1.1:.2f}): {(final_prices > current_price * 1.1).mean():.2%}")
    print(f"Probability of price 10% lower (<${current_price * 0.9:.2f}): {(final_prices < current_price * 0.9).mean():.2%}")
    
    # Calculate expected percentage return based on the simulations
    expected_return = (mean_final / current_price - 1) * 100
    print(f"\nExpected return over {days_to_simulate} days: {expected_return:+.2f}%")
    print(f"Annualized expected return: {expected_return * 365 / days_to_simulate:+.2f}%")
    
    return fig

def plot_price_simulations(simulation_results, num_paths_to_plot=5, benchmark_results=None, save_path=None):
    """
    Plot the simulated price paths
    
    Args:
        simulation_results (dict): Results from simulate_geometric_brownian_motion
        num_paths_to_plot (int, optional): Number of individual paths to plot. Defaults to 5.
        benchmark_results (dict, optional): Benchmark simulation results for comparison
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Extract data from simulation results
    ticker = simulation_results['ticker']
    current_price = simulation_results['current_price']
    date_range = simulation_results['date_range']
    price_paths = simulation_results['price_paths']
    mean_path = simulation_results['mean_path']
    percentile_5 = simulation_results['percentile_5']
    percentile_95 = simulation_results['percentile_95']
    historical_volatility = simulation_results['historical_volatility']
    
    # Normalize paths for percentage change view
    normalized_paths = price_paths / current_price
    normalized_mean = mean_path / current_price
    normalized_p5 = percentile_5 / current_price
    normalized_p95 = percentile_95 / current_price
    
    # Create a figure with two subplots if benchmark is provided, otherwise just one
    if benchmark_results:
        fig = plt.figure(num="Stock Price Simulation", figsize=(18, 10))
        ax1 = fig.add_subplot(121)  # Absolute price plot
        ax2 = fig.add_subplot(122)  # Relative return plot
    else:
        fig = plt.figure(num="Stock Price Simulation", figsize=(12, 8))
        ax1 = fig.add_subplot(111)  # Just the main plot
        ax2 = None
    
    # Plot a subset of individual paths with low opacity
    num_simulations = price_paths.shape[0]
    indices_to_plot = np.random.choice(num_simulations, min(num_paths_to_plot, num_simulations), replace=False)
    
    for i in indices_to_plot:
        ax1.plot(date_range, price_paths[i, :], alpha=0.3, linewidth=0.8)
    
    # Plot mean path
    ax1.plot(date_range, mean_path, 'b-', linewidth=2, label=f'{ticker} Mean Path')
    
    # Plot confidence interval (5th to 95th percentile)
    ax1.fill_between(date_range, percentile_5, percentile_95, color='b', alpha=0.2, label=f'{ticker} 90% Confidence')
    
    # Plot current price point
    ax1.plot(date_range[0], current_price, 'ro', markersize=8, label=f'Current Price: ${current_price:.2f}')
    
    # Add horizontal line at current price
    ax1.axhline(y=current_price, color='r', linestyle='--', alpha=0.5)
    
    # Plot benchmark if provided
    if benchmark_results and ax2 is not None:
        benchmark_ticker = benchmark_results['ticker']
        benchmark_current = benchmark_results['current_price']
        benchmark_paths = benchmark_results['price_paths']
        benchmark_mean = benchmark_results['mean_path']
        benchmark_p5 = benchmark_results['percentile_5']
        benchmark_p95 = benchmark_results['percentile_95']
        
        # Normalize benchmark paths
        norm_benchmark_paths = benchmark_paths / benchmark_current
        norm_benchmark_mean = benchmark_mean / benchmark_current
        norm_benchmark_p5 = benchmark_p5 / benchmark_current
        norm_benchmark_p95 = benchmark_p95 / benchmark_current
        
        # Plot normalized returns for both the stock and benchmark
        ax2.plot(date_range, normalized_mean, 'b-', linewidth=2, label=f'{ticker} Return')
        ax2.plot(date_range, norm_benchmark_mean, 'g-', linewidth=2, label=f'{benchmark_ticker} Return')
        
        # Plot confidence intervals
        ax2.fill_between(date_range, normalized_p5, normalized_p95, color='b', alpha=0.1)
        ax2.fill_between(date_range, norm_benchmark_p5, norm_benchmark_p95, color='g', alpha=0.1)
        
        # Add horizontal line at 100% (no change)
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        
        # Set titles and labels for relative return plot
        ax2.set_title(f"Relative Returns: {ticker} vs {benchmark_ticker}")
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Relative Return (Initial = 1.0)')
        
        # Format x-axis for dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid and legend
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
    
    # Set title and labels for main plot
    ax1.set_title(f"{ticker} Stock Price Simulation\nGeometric Brownian Motion (Historical Vol: {historical_volatility:.2%})")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price ($)')
    
    # Format x-axis to show dates nicely
    num_dates = min(10, len(date_range))
    date_indices = np.linspace(0, len(date_range)-1, num_dates, dtype=int)
    ax1.set_xticks([date_range[i] for i in date_indices])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Add grid and legend
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Add annotations explaining the simulation
    textstr = '\n'.join([
        f'Simulated using Geometric Brownian Motion',
        f'Total paths: {price_paths.shape[0]}',
        f'Historical volatility: {historical_volatility:.2%}',
        f'Risk-free rate: {simulation_results["risk_free_rate"]:.2%}'
    ])
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    fig.tight_layout()
    
    # Save or display the plot
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Simulate stock price using Geometric Brownian Motion')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--days', type=int, default=180, help='Number of days to simulate (default: 180)')
    parser.add_argument('--simulations', type=int, default=1000, help='Number of simulation paths (default: 1000)')
    parser.add_argument('--benchmark', type=str, help='Benchmark ticker for comparison (e.g., SPY)')
    parser.add_argument('--save', type=str, help='Path to save the plot (optional)')
    
    args = parser.parse_args()
    
    # Run the simulation
    fig = run_simulation(
        ticker=args.ticker,
        days_to_simulate=args.days,
        num_simulations=args.simulations,
        benchmark=args.benchmark,
        save_path=args.save
    )
    
    # Show the plot if not saving
    if not args.save:
        plt.show()

if __name__ == "__main__":
    main() 