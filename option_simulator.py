#!/usr/bin/env python3
"""
Option Simulator

This module simulates the PnL of options trades using the Black-Scholes model.
It calculates a 2D matrix of PnLs for a range of stock prices and dates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from scipy.stats import norm
from datetime import datetime, timedelta
import yfinance as yf
from option_fetcher import fetch_option_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option price using Black-Scholes model
    
    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free interest rate
        sigma (float): Volatility of the underlying asset
        option_type (str): 'call' or 'put'
    
    Returns:
        float: Option price
    """
    # Ensure T is positive to avoid math domain errors
    T = max(T, 1e-10)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def calculate_option_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option greeks using Black-Scholes model
    
    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free interest rate
        sigma (float): Volatility of the underlying asset
        option_type (str): 'call' or 'put'
    
    Returns:
        dict: Option greeks (delta, gamma, theta, vega)
    """
    # Ensure T is positive to avoid math domain errors
    T = max(T, 1e-10)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate delta
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
    else:  # put
        delta = norm.cdf(d1) - 1
    
    # Calculate gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Calculate theta
    if option_type.lower() == 'call':
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    # Convert theta to daily (from annual)
    theta = theta / 365
    
    # Calculate vega (same for calls and puts)
    vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01  # 1% change in volatility
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }

def get_risk_free_rate():
    """
    Get current risk-free interest rate (using 3-month Treasury yield as proxy)
    
    Returns:
        float: Risk-free rate as a decimal (e.g., 0.05 for 5%)
    """
    try:
        # Fetch 3-month Treasury yield data
        treasury = yf.Ticker("^IRX")
        current_rate = treasury.history(period="1d")['Close'].iloc[-1] / 100
        return current_rate
    except Exception as e:
        logger.warning(f"Could not fetch current risk-free rate: {str(e)}")
        logger.warning("Using default risk-free rate of 3%")
        return 0.03  # Default to 3% if unable to fetch

def simulate_option_pnl(
    ticker, 
    option_type, 
    expiry_date, 
    target_delta, 
    position_type='buy',
    num_contracts=1
):
    """
    Simulate the PnL of an options trade using the Black-Scholes model
    
    Args:
        ticker (str): Ticker symbol
        option_type (str): 'call' or 'put'
        expiry_date (str): Expiry date in 'YYYY-MM-DD' format
        target_delta (float): Target delta value
        position_type (str, optional): 'buy' or 'sell'. Defaults to 'buy'.
        num_contracts (int, optional): Number of contracts. Defaults to 1.
    
    Returns:
        dict: Simulation results including PnL matrix and trade details
    """
    # Validate inputs
    if position_type.lower() not in ['buy', 'sell']:
        raise ValueError("position_type must be either 'buy' or 'sell'")
    
    # Set start date to today
    start_date = datetime.now().strftime('%Y-%m-%d')
    
    # Parse dates
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    expiry_date_dt = datetime.strptime(expiry_date, '%Y-%m-%d').date()
    
    # Ensure start date is not after expiry date
    if start_date_dt > expiry_date_dt:
        raise ValueError("Current date is after expiry date. Please choose a future expiry date.")
    
    # Fetch option data
    logger.info(f"Fetching option data for {ticker} {option_type} with target delta {target_delta}")
    option_data = fetch_option_data(
        ticker=ticker,
        option_type=option_type,
        target_expiry_date=expiry_date,
        target_delta=target_delta
    )
    
    # Extract key information
    actual_expiry_date = option_data['expiry_date']
    actual_expiry_date_dt = datetime.strptime(actual_expiry_date, '%Y-%m-%d').date()
    strike_price = option_data['strike']
    current_price = option_data['underlying_price']
    option_price = option_data['option_data']['lastPrice']
    implied_volatility = option_data['option_data']['impliedVolatility']
    
    # Get risk-free rate
    risk_free_rate = get_risk_free_rate()
    
    # Calculate days to expiry from start date
    days_to_expiry = (actual_expiry_date_dt - start_date_dt).days
    
    # Generate date range from start date to expiry date
    date_range = [start_date_dt + timedelta(days=i) for i in range(days_to_expiry + 1)]
    
    # Calculate price range based on implied volatility and 2-standard deviation move
    # Standard deviation of price at expiry = S * σ * sqrt(T)
    time_to_expiry_years = days_to_expiry / 365.0
    std_dev = current_price * implied_volatility * np.sqrt(time_to_expiry_years)
    
    # Use 2-standard deviations for the price range (approximately 95% confidence interval)
    min_price = max(current_price - 2 * std_dev, 0.01)  # Ensure price is positive
    max_price = current_price + 2 * std_dev
    
    logger.info(f"Price range based on 2-standard deviations: ${min_price:.2f} to ${max_price:.2f}")
    
    # Generate price range
    price_range = np.linspace(min_price, max_price, 50)
    
    # Create empty matrices for PnL and option values
    pnl_matrix = np.zeros((len(date_range), len(price_range)))
    option_value_matrix = np.zeros((len(date_range), len(price_range)))
    
    # Calculate PnL for each combination of date and price
    for i, date in enumerate(date_range):
        # Calculate time to expiry in years
        time_to_expiry = (actual_expiry_date_dt - date).days / 365.0
        
        for j, price in enumerate(price_range):
            # Calculate option price using Black-Scholes
            option_value = black_scholes_price(
                S=price,
                K=strike_price,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=implied_volatility,
                option_type=option_type
            )
            
            option_value_matrix[i, j] = option_value
            
            # Calculate PnL based on position type
            if position_type.lower() == 'buy':
                # For buying options, PnL = (current value - initial price) * 100 * num_contracts
                pnl_matrix[i, j] = (option_value - option_price) * 100 * num_contracts
            else:  # sell
                # For selling options, PnL = (initial price - current value) * 100 * num_contracts
                pnl_matrix[i, j] = (option_price - option_value) * 100 * num_contracts
    
    # Calculate greeks at the start
    initial_greeks = calculate_option_greeks(
        S=current_price,
        K=strike_price,
        T=days_to_expiry / 365.0,
        r=risk_free_rate,
        sigma=implied_volatility,
        option_type=option_type
    )
    
    # Return results
    return {
        'ticker': ticker,
        'option_type': option_type,
        'position_type': position_type,
        'strike_price': strike_price,
        'current_price': current_price,
        'option_price': option_price,
        'implied_volatility': implied_volatility,
        'start_date': start_date,
        'expiry_date': actual_expiry_date,
        'days_to_expiry': days_to_expiry,
        'num_contracts': num_contracts,
        'contract_value': option_price * 100 * num_contracts,
        'max_profit': np.max(pnl_matrix),
        'max_loss': np.min(pnl_matrix),
        'date_range': date_range,
        'price_range': price_range,
        'pnl_matrix': pnl_matrix,
        'option_value_matrix': option_value_matrix,
        'initial_greeks': initial_greeks,
        'risk_free_rate': risk_free_rate
    }

def plot_pnl_heatmap(simulation_results, save_path=None):
    """
    Plot a heatmap of the PnL matrix
    
    Args:
        simulation_results (dict): Results from simulate_option_pnl
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    # Extract data from simulation results
    ticker = simulation_results['ticker']
    option_type = simulation_results['option_type']
    position_type = simulation_results['position_type']
    strike_price = simulation_results['strike_price']
    current_price = simulation_results['current_price']
    date_range = simulation_results['date_range']
    price_range = simulation_results['price_range']
    pnl_matrix = simulation_results['pnl_matrix']
    option_price = simulation_results['option_price']
    num_contracts = simulation_results['num_contracts']
    
    # Calculate initial investment
    contract_multiplier = 100  # Each contract represents 100 shares
    initial_investment = option_price * contract_multiplier * num_contracts
    
    # Convert PnL matrix to percentage of initial investment
    pnl_percent_matrix = (pnl_matrix / initial_investment) * 100
    
    # Create figure and axes with a specific figure number
    fig = plt.figure(num="PnL Heatmap", figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Transpose the PnL matrix to flip axes (price on y-axis, date on x-axis)
    pnl_matrix_t = pnl_percent_matrix.T
    
    # Create heatmap with flipped axes
    im = ax.imshow(
        pnl_matrix_t, 
        aspect='auto', 
        origin='lower', 
        cmap='RdYlGn',
        extent=[0, len(date_range)-1, price_range[0], price_range[-1]],
        vmin=min(-100, np.min(pnl_percent_matrix)),  # Ensure color scale is balanced for percentage values
        vmax=max(100, np.max(pnl_percent_matrix))
    )
    
    # Add colorbar
    cbar = fig.colorbar(im)
    cbar.set_label('PnL (%)')
    
    # Set title and labels
    ax.set_title(f"{position_type.capitalize()} {ticker} {option_type.upper()} Option PnL Simulation\nStrike: ${strike_price:.2f}, Current Price: ${current_price:.2f}")
    ax.set_ylabel('Stock Price ($)')
    ax.set_xlabel('Date')
    
    # Set x-ticks to show dates
    num_dates = min(10, len(date_range))  # Limit to 10 dates to avoid overcrowding
    date_indices = np.linspace(0, len(date_range)-1, num_dates, dtype=int)
    ax.set_xticks(date_indices)
    ax.set_xticklabels([date_range[i].strftime('%Y-%m-%d') for i in date_indices], rotation=45, ha='right')
    
    # Add strike price line (horizontal now)
    ax.axhline(y=strike_price, color='black', linestyle='--', alpha=0.7, label=f'Strike: ${strike_price:.2f}')
    
    # Add current price line (horizontal now)
    ax.axhline(y=current_price, color='blue', linestyle='-', alpha=0.7, label=f'Current: ${current_price:.2f}')
    
    # Add break-even line(s) - now need to find for each date column
    for i in range(len(date_range)):
        # Find where PnL crosses zero for this date
        for j in range(1, len(price_range)):
            if (pnl_percent_matrix[i, j-1] < 0 and pnl_percent_matrix[i, j] >= 0) or (pnl_percent_matrix[i, j-1] >= 0 and pnl_percent_matrix[i, j] < 0):
                # Interpolate to find the exact break-even price
                p1, p2 = price_range[j-1], price_range[j]
                pnl1, pnl2 = pnl_percent_matrix[i, j-1], pnl_percent_matrix[i, j]
                break_even = p1 + (p2 - p1) * (-pnl1) / (pnl2 - pnl1)
                
                # Only plot break-even for start date and expiry date to avoid clutter
                if i == 0:
                    ax.plot(i, break_even, 'ro', markersize=5)
                    ax.annotate(f'${break_even:.2f}', (i, break_even), xytext=(5, 5), 
                                textcoords='offset points', color='red')
                elif i == len(date_range) - 1:
                    ax.plot(i, break_even, 'ro', markersize=5)
                    ax.annotate(f'${break_even:.2f}', (i, break_even), xytext=(5, 5), 
                                textcoords='offset points', color='red')
    
    ax.legend()
    fig.tight_layout()
    
    # Save or display the plot
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
        
    return fig

def plot_pnl_slices(simulation_results, save_path=None):
    """
    Plot PnL slices at different dates
    
    Args:
        simulation_results (dict): Results from simulate_option_pnl
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    # Extract data from simulation results
    ticker = simulation_results['ticker']
    option_type = simulation_results['option_type']
    position_type = simulation_results['position_type']
    strike_price = simulation_results['strike_price']
    current_price = simulation_results['current_price']
    date_range = simulation_results['date_range']
    price_range = simulation_results['price_range']
    pnl_matrix = simulation_results['pnl_matrix']
    
    # Create figure and axes with a specific figure number
    fig = plt.figure(num="PnL Slices", figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Select dates to plot (start, 25%, 50%, 75%, expiry)
    date_indices = [
        0,  # Start date
        len(date_range) // 4,  # 25% through
        len(date_range) // 2,  # 50% through
        3 * len(date_range) // 4,  # 75% through
        -1  # Expiry date
    ]
    
    # Plot PnL slices for selected dates
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for i, idx in enumerate(date_indices):
        date = date_range[idx]
        ax.plot(
            price_range, 
            pnl_matrix[idx], 
            color=colors[i], 
            label=f"{date.strftime('%Y-%m-%d')} ({(date - date_range[0]).days} days)"
        )
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add strike price line
    ax.axvline(x=strike_price, color='black', linestyle='--', alpha=0.7, label=f'Strike: ${strike_price:.2f}')
    
    # Add current price line
    ax.axvline(x=current_price, color='blue', linestyle=':', alpha=0.7, label=f'Current: ${current_price:.2f}')
    
    # Set title and labels
    ax.set_title(f"{position_type.capitalize()} {ticker} {option_type.upper()} Option PnL Slices\nStrike: ${strike_price:.2f}, Current Price: ${current_price:.2f}")
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel('PnL ($)')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    fig.tight_layout()
    
    # Save or display the plot
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def generate_simulation_report(simulation_results):
    """
    Generate a text report of the simulation results
    
    Args:
        simulation_results (dict): Results from simulate_option_pnl
    
    Returns:
        str: Text report
    """
    # Extract data from simulation results
    ticker = simulation_results['ticker']
    option_type = simulation_results['option_type']
    position_type = simulation_results['position_type']
    strike_price = simulation_results['strike_price']
    current_price = simulation_results['current_price']
    option_price = simulation_results['option_price']
    implied_volatility = simulation_results['implied_volatility']
    start_date = simulation_results['start_date']
    expiry_date = simulation_results['expiry_date']
    days_to_expiry = simulation_results['days_to_expiry']
    num_contracts = simulation_results['num_contracts']
    contract_value = simulation_results['contract_value']
    max_profit = simulation_results['max_profit']
    max_loss = simulation_results['max_loss']
    initial_greeks = simulation_results['initial_greeks']
    risk_free_rate = simulation_results['risk_free_rate']
    
    # Generate report
    report = [
        "=" * 80,
        f"{'OPTION TRADE SIMULATION REPORT':^80}",
        "=" * 80,
        "",
        f"Trade Setup:",
        f"  - Ticker: {ticker}",
        f"  - Position: {position_type.capitalize()} {option_type.upper()} Option",
        f"  - Strike Price: ${strike_price:.2f}",
        f"  - Current Stock Price: ${current_price:.2f}",
        f"  - Option Price: ${option_price:.2f} per share (${option_price * 100:.2f} per contract)",
        f"  - Implied Volatility: {implied_volatility:.2%}",
        f"  - Risk-Free Rate: {risk_free_rate:.2%}",
        f"  - Number of Contracts: {num_contracts}",
        f"  - Total Investment: ${contract_value:.2f}",
        "",
        f"Dates:",
        f"  - Start Date: {start_date}",
        f"  - Expiry Date: {expiry_date}",
        f"  - Days to Expiry: {days_to_expiry}",
        "",
        f"Initial Greeks:",
        f"  - Delta: {initial_greeks['delta']:.4f}",
        f"  - Gamma: {initial_greeks['gamma']:.4f}",
        f"  - Theta: ${initial_greeks['theta']:.4f} per day",
        f"  - Vega: ${initial_greeks['vega']:.4f} per 1% change in volatility",
        "",
        f"Potential Outcomes:",
        f"  - Maximum Profit: ${max_profit:.2f}",
        f"  - Maximum Loss: ${max_loss:.2f}",
        f"  - Risk/Reward Ratio: {abs(max_loss / max_profit):.2f}" if max_profit != 0 else "  - Risk/Reward Ratio: N/A",
        "",
        "=" * 80
    ]
    
    return "\n".join(report)

if __name__ == "__main__":
    # Example usage
    results = simulate_option_pnl(
        ticker="AAPL",
        option_type="call",
        expiry_date="2023-12-15",
        target_delta=0.5,
        position_type="buy",
        num_contracts=1
    )
    
    # Print report
    print(generate_simulation_report(results))
    
    # Plot results
    plot_pnl_heatmap(results)
    plot_pnl_slices(results)
    
    # Show all plots
    plt.show() 