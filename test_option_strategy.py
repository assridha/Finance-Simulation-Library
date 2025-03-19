#!/usr/bin/env python3
"""
Test Option Strategy

This script tests the OptionStrategy class functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from option_simulator import OptionStrategy, black_scholes_price, get_risk_free_rate
from datetime import datetime, timedelta

def test_option_strategy_1d():
    """
    Test OptionStrategy with 1D price array
    """
    print("Testing OptionStrategy with 1D price array...")
    
    # Create strategy
    ticker = "TEST"
    option_type = "call"
    strike_price = 100.0
    current_price = 100.0
    option_price = 5.0
    implied_volatility = 0.2
    risk_free_rate = 0.03
    position_type = "buy"
    num_contracts = 1
    
    # Set expiry to 30 days from now
    expiry_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Create option strategy
    strategy = OptionStrategy(
        ticker=ticker,
        option_type=option_type,
        strike_price=strike_price,
        expiry_date=expiry_date,
        current_price=current_price,
        option_price=option_price,
        implied_volatility=implied_volatility,
        position_type=position_type,
        num_contracts=num_contracts,
        risk_free_rate=risk_free_rate
    )
    
    # Create array of stock prices
    price_range = np.linspace(80, 120, 41)
    
    # Calculate option values and PnL
    time_to_expiry = strategy.days_to_expiry / 365.0
    option_values = strategy.calculate_option_value(price_range, time_to_expiry)
    pnl = strategy.calculate_pnl(price_range)
    
    # Verify dimensions
    print(f"Price range shape: {price_range.shape}")
    print(f"Option values shape: {option_values.shape}")
    print(f"PnL shape: {pnl.shape}")
    
    # Verify values at specific points
    atm_index = np.abs(price_range - strike_price).argmin()
    print(f"At strike price {strike_price}:")
    print(f"  - Option value: ${option_values[atm_index]:.2f}")
    print(f"  - PnL: ${pnl[atm_index]:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(price_range, option_values)
    plt.axvline(x=strike_price, linestyle='--', color='red', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.title('Option Values')
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Option Value ($)')
    
    plt.subplot(1, 2, 2)
    plt.plot(price_range, pnl)
    plt.axvline(x=strike_price, linestyle='--', color='red', alpha=0.5)
    plt.axhline(y=0, linestyle='-', color='black', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('PnL')
    plt.xlabel('Stock Price ($)')
    plt.ylabel('PnL ($)')
    
    plt.tight_layout()
    plt.show()
    
    return strategy, price_range, pnl

def test_option_strategy_2d():
    """
    Test OptionStrategy with 2D price array
    """
    print("\nTesting OptionStrategy with 2D price array...")
    
    # Create strategy
    ticker = "TEST"
    option_type = "call"
    strike_price = 100.0
    current_price = 100.0
    option_price = 5.0
    implied_volatility = 0.2
    risk_free_rate = 0.03
    position_type = "buy"
    num_contracts = 1
    
    # Set expiry to 30 days from now
    expiry_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Create option strategy
    strategy = OptionStrategy(
        ticker=ticker,
        option_type=option_type,
        strike_price=strike_price,
        expiry_date=expiry_date,
        current_price=current_price,
        option_price=option_price,
        implied_volatility=implied_volatility,
        position_type=position_type,
        num_contracts=num_contracts,
        risk_free_rate=risk_free_rate
    )
    
    # Create array of stock prices (2D grid)
    price_range = np.linspace(80, 120, 41)
    date_range = [datetime.now().date() + timedelta(days=i) for i in range(0, strategy.days_to_expiry + 1, 5)]
    num_dates = len(date_range)
    
    # Create a 2D price grid
    price_grid = np.zeros((num_dates, len(price_range)))
    for i in range(num_dates):
        price_grid[i, :] = price_range
    
    # Create times to expiry
    times_to_expiry = np.array([(strategy.expiry_date_dt - date).days / 365.0 for date in date_range])
    
    # Calculate PnL matrix
    pnl_matrix = strategy.calculate_pnl(price_grid, times_to_expiry)
    
    # Verify dimensions
    print(f"Price grid shape: {price_grid.shape}")
    print(f"PnL matrix shape: {pnl_matrix.shape}")
    
    # Plot results as heatmap and slices
    plt.figure(figsize=(15, 6))
    
    # Heatmap
    plt.subplot(1, 2, 1)
    plt.imshow(
        pnl_matrix,
        aspect='auto',
        origin='lower',
        extent=[price_range[0], price_range[-1], 0, len(date_range) - 1],
        cmap='RdYlGn'
    )
    plt.colorbar(label='PnL ($)')
    plt.axvline(x=strike_price, linestyle='--', color='black', alpha=0.5)
    plt.title('PnL Heatmap')
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Date Index')
    
    # Date ticks
    date_ticks = np.arange(len(date_range))
    plt.yticks(date_ticks, [date.strftime('%m-%d') for date in date_range])
    
    # PnL slices for different dates
    plt.subplot(1, 2, 2)
    
    # Select a few dates to plot
    date_indices = [0, len(date_range) // 2, -1]  # First, middle, last
    colors = ['blue', 'green', 'red']
    
    for i, date_idx in enumerate(date_indices):
        plt.plot(
            price_range, 
            pnl_matrix[date_idx], 
            color=colors[i], 
            label=f"{date_range[date_idx].strftime('%Y-%m-%d')} ({times_to_expiry[date_idx]:.2f} years)"
        )
    
    plt.axvline(x=strike_price, linestyle='--', color='black', alpha=0.5)
    plt.axhline(y=0, linestyle='-', color='black', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('PnL Slices')
    plt.xlabel('Stock Price ($)')
    plt.ylabel('PnL ($)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return strategy, price_grid, pnl_matrix

def compare_with_original():
    """
    Compare new method with original approach
    """
    print("\nComparing new method with original approach...")
    
    # Create a simple case
    ticker = "TEST"
    option_type = "call"
    strike_price = 100.0
    current_price = 100.0
    option_price = 5.0
    implied_volatility = 0.2
    risk_free_rate = 0.03
    position_type = "buy"
    num_contracts = 1
    
    # Set expiry to 30 days from now
    days_to_expiry = 30
    expiry_date = (datetime.now() + timedelta(days=days_to_expiry)).strftime('%Y-%m-%d')
    
    # Create option strategy
    strategy = OptionStrategy(
        ticker=ticker,
        option_type=option_type,
        strike_price=strike_price,
        expiry_date=expiry_date,
        current_price=current_price,
        option_price=option_price,
        implied_volatility=implied_volatility,
        position_type=position_type,
        num_contracts=num_contracts,
        risk_free_rate=risk_free_rate
    )
    
    # Create array of stock prices
    price_range = np.linspace(80, 120, 41)
    
    # Original approach - calculate PnL for a single date
    time_to_expiry = days_to_expiry / 365.0
    option_values_original = np.zeros_like(price_range)
    pnl_original = np.zeros_like(price_range)
    
    for i, price in enumerate(price_range):
        option_value = black_scholes_price(
            S=price,
            K=strike_price,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=implied_volatility,
            option_type=option_type
        )
        option_values_original[i] = option_value
        
        if position_type.lower() == 'buy':
            pnl_original[i] = (option_value - option_price) * 100 * num_contracts
        else:  # sell
            pnl_original[i] = (option_price - option_value) * 100 * num_contracts
    
    # New approach
    pnl_new = strategy.calculate_pnl(price_range)
    
    # Compare
    diff = np.abs(pnl_original - pnl_new)
    max_diff = np.max(diff)
    print(f"Maximum difference between original and new methods: ${max_diff:.2f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(price_range, pnl_original, 'b-', label='Original Method')
    plt.plot(price_range, pnl_new, 'r--', label='New Method')
    plt.axvline(x=strike_price, linestyle='--', color='black', alpha=0.5)
    plt.axhline(y=0, linestyle='-', color='black', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('PnL Comparison')
    plt.xlabel('Stock Price ($)')
    plt.ylabel('PnL ($)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return max_diff

if __name__ == "__main__":
    # Test 1D option strategy
    strategy_1d, price_range_1d, pnl_1d = test_option_strategy_1d()
    
    # Test 2D option strategy
    strategy_2d, price_grid_2d, pnl_matrix_2d = test_option_strategy_2d()
    
    # Compare with original approach
    max_diff = compare_with_original()
    
    if max_diff < 1e-10:
        print("SUCCESS: New method matches original method!")
    else:
        print(f"WARNING: New method differs from original method (max diff: ${max_diff:.2f})") 