#!/usr/bin/env python3
"""
Stock Price Simulation Examples

This script demonstrates how to use the stock price simulator.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from financial_sim_library.stock_simulator.models.gbm import GBMModel
from financial_sim_library.visualization.price_plots import plot_price_simulations, plot_price_distribution, plot_price_heatmap

def basic_simulation():
    """Run a basic stock price simulation with default parameters."""
    print("Running basic stock price simulation...")
    
    # Create GBM model for AAPL
    model = GBMModel(ticker="AAPL")
    
    # Run simulation
    results = model.simulate(
        days_to_simulate=30,
        num_simulations=1000
    )
    
    # Print summary statistics
    stats = results['statistics']
    print(f"Simulation results for {results['ticker']}:")
    print(f"Current price: ${results['current_price']:.2f}")
    print(f"Mean final price: ${stats['mean']:.2f}")
    print(f"Median final price: ${stats['median']:.2f}")
    print(f"Min final price: ${stats['min']:.2f}")
    print(f"Max final price: ${stats['max']:.2f}")
    print(f"Expected return: {stats['expected_return']:.2f}%")
    print(f"Probability of price increase: {stats['prob_above_current']:.2f}%")
    print(f"Probability of >10% increase: {stats['prob_10pct_up']:.2f}%")
    print(f"Probability of >10% decrease: {stats['prob_10pct_down']:.2f}%")
    
    # Plot results
    plot_price_simulations(results)
    
    return results

def custom_simulation(ticker, days_to_simulate=60, num_simulations=500):
    """
    Run a custom stock price simulation.
    
    Args:
        ticker (str): The ticker to simulate
        days_to_simulate (int, optional): Number of days to simulate. Defaults to 60.
        num_simulations (int, optional): Number of simulation paths. Defaults to 500.
    
    Returns:
        dict: Simulation results
    """
    print(f"Running custom simulation for {ticker}...")
    
    # Create GBM model
    model = GBMModel(ticker=ticker)
    
    # Run simulation
    results = model.simulate(
        days_to_simulate=days_to_simulate,
        num_simulations=num_simulations
    )
    
    # Plot results
    plot_price_simulations(results, num_paths_to_plot=10)
    plot_price_distribution(results)
    plot_price_heatmap(results)
    
    return results

def compare_volatilities(ticker="SPY", volatilities=[0.1, 0.2, 0.3, 0.4]):
    """
    Compare simulations with different volatilities.
    
    Args:
        ticker (str, optional): The ticker to simulate. Defaults to "SPY".
        volatilities (list, optional): List of volatilities to compare. Defaults to [0.1, 0.2, 0.3, 0.4].
    """
    print(f"Comparing volatilities for {ticker}...")
    
    # Fetch current price
    base_model = GBMModel(ticker=ticker)
    current_price = base_model.current_price
    risk_free_rate = base_model.risk_free_rate
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, vol in enumerate(volatilities):
        # Create model with specific volatility
        model = GBMModel(
            ticker=ticker,
            current_price=current_price,
            volatility=vol,
            risk_free_rate=risk_free_rate
        )
        
        # Run simulation
        results = model.simulate(
            days_to_simulate=90,
            num_simulations=500
        )
        
        # Get final prices
        final_prices = results['price_paths'][:, -1]
        
        # Plot histogram on the appropriate subplot
        ax = axes[i]
        ax.hist(final_prices, bins=50, alpha=0.7)
        ax.axvline(x=current_price, color='r', linestyle='--', label=f'Current: ${current_price:.2f}')
        ax.axvline(x=np.mean(final_prices), color='g', linestyle='-', label=f'Mean: ${np.mean(final_prices):.2f}')
        
        # Calculate expected return and probability of increase
        expected_return = (np.mean(final_prices) / current_price - 1) * 100
        prob_increase = np.mean(final_prices > current_price) * 100
        
        # Add title and labels
        ax.set_title(f"Volatility: {vol:.1%}, Expected Return: {expected_return:.2f}%, P(Increase): {prob_increase:.1f}%")
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def simulate_multiple_tickers(tickers=["AAPL", "MSFT", "AMZN", "GOOGL"], days=90):
    """
    Simulate multiple tickers and compare expected returns.
    
    Args:
        tickers (list, optional): List of tickers to simulate. Defaults to ["AAPL", "MSFT", "AMZN", "GOOGL"].
        days (int, optional): Days to simulate. Defaults to 90.
    
    Returns:
        dict: Dictionary mapping tickers to simulation results
    """
    print(f"Simulating multiple tickers: {tickers}")
    
    results = {}
    expected_returns = []
    
    for ticker in tickers:
        # Create model
        model = GBMModel(ticker=ticker)
        
        # Run simulation
        sim_results = model.simulate(
            days_to_simulate=days,
            num_simulations=1000
        )
        
        # Store results
        results[ticker] = sim_results
        expected_returns.append(sim_results['statistics']['expected_return'])
    
    # Plot expected returns comparison
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, expected_returns)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f"Expected {days}-Day Returns")
    plt.xlabel("Ticker")
    plt.ylabel("Expected Return (%)")
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run examples
    print("\n" + "="*80)
    print("STOCK PRICE SIMULATION EXAMPLES")
    print("="*80 + "\n")
    
    # Example 1: Basic simulation
    basic_results = basic_simulation()
    
    # Example 2: Custom simulation
    custom_results = custom_simulation("TSLA", days_to_simulate=90, num_simulations=1000)
    
    # Example 3: Compare volatilities
    compare_volatilities()
    
    # Example 4: Simulate multiple tickers
    multi_results = simulate_multiple_tickers()
    
    print("\nAll examples completed successfully!") 