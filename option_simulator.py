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
import math

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

def calculate_historical_volatility(ticker, lookback_days=30):
    """
    Calculate historical volatility from the last month of price data
    
    Args:
        ticker (str): Ticker symbol
        lookback_days (int, optional): Number of days to look back. Defaults to 30.
    
    Returns:
        float: Annualized historical volatility
    """
    try:
        # Add some buffer days to account for weekends and holidays
        buffer_days = int(lookback_days * 1.5)
        
        # Fetch historical data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=f"{buffer_days}d")
        
        # Ensure we have enough data
        if len(hist_data) < lookback_days:
            logger.warning(f"Insufficient historical data for {ticker}. Using available data ({len(hist_data)} days).")
        
        # Calculate daily returns
        hist_data['Return'] = hist_data['Close'].pct_change()
        
        # Drop NaN values
        hist_data = hist_data.dropna()
        
        # Use the most recent data up to lookback_days
        if len(hist_data) > lookback_days:
            hist_data = hist_data.tail(lookback_days)
        
        # Calculate daily standard deviation
        daily_std = hist_data['Return'].std()
        
        # Annualize (approximately 252 trading days in a year)
        annual_volatility = daily_std * math.sqrt(252)
        
        logger.info(f"Calculated historical volatility for {ticker}: {annual_volatility:.2%}")
        return annual_volatility
    except Exception as e:
        logger.error(f"Error calculating historical volatility: {str(e)}")
        logger.warning("Using default historical volatility of 30%")
        return 0.30  # Default to 30% if unable to fetch or calculate

class OptionStrategy:
    """
    Class for creating and evaluating option trading strategies.
    Currently supports single-leg strategies (buy/sell call/put).
    """
    
    def __init__(self, 
                 ticker, 
                 option_type, 
                 strike_price, 
                 expiry_date, 
                 current_price,
                 option_price,
                 implied_volatility,
                 position_type='buy',
                 num_contracts=1,
                 risk_free_rate=None):
        """
        Initialize option strategy
        
        Args:
            ticker (str): Ticker symbol
            option_type (str): 'call' or 'put'
            strike_price (float): Option strike price
            expiry_date (str): Expiry date in 'YYYY-MM-DD' format
            current_price (float): Current price of the underlying asset
            option_price (float): Current price of the option
            implied_volatility (float): Implied volatility of the option
            position_type (str, optional): 'buy' or 'sell'. Defaults to 'buy'.
            num_contracts (int, optional): Number of contracts. Defaults to 1.
            risk_free_rate (float, optional): Risk-free rate. If None, will be fetched.
        """
        # Validate inputs
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be either 'call' or 'put'")
        if position_type.lower() not in ['buy', 'sell']:
            raise ValueError("position_type must be either 'buy' or 'sell'")
        
        self.ticker = ticker
        self.option_type = option_type.lower()
        self.strike_price = strike_price
        self.expiry_date = expiry_date
        self.current_price = current_price
        self.option_price = option_price
        self.implied_volatility = implied_volatility
        self.position_type = position_type.lower()
        self.num_contracts = num_contracts
        
        # Parse expiry date
        self.expiry_date_dt = datetime.strptime(expiry_date, '%Y-%m-%d').date()
        
        # Set start date to today and calculate days to expiry
        self.start_date = datetime.now().date()
        self.days_to_expiry = (self.expiry_date_dt - self.start_date).days
        
        # Get risk-free rate if not provided
        self.risk_free_rate = risk_free_rate if risk_free_rate is not None else get_risk_free_rate()
        
        # Calculate contract value
        self.contract_value = option_price * 100 * num_contracts
        
        # Calculate initial greeks
        self.initial_greeks = calculate_option_greeks(
            S=current_price,
            K=strike_price,
            T=self.days_to_expiry / 365.0,
            r=self.risk_free_rate,
            sigma=implied_volatility,
            option_type=option_type
        )
    
    def calculate_option_value(self, stock_price, time_to_expiry):
        """
        Calculate option value using Black-Scholes model
        
        Args:
            stock_price (float or np.ndarray): Stock price(s)
            time_to_expiry (float): Time to expiry in years
            
        Returns:
            float or np.ndarray: Option value(s)
        """
        return black_scholes_price(
            S=stock_price,
            K=self.strike_price,
            T=time_to_expiry,
            r=self.risk_free_rate,
            sigma=self.implied_volatility,
            option_type=self.option_type
        )
    
    def calculate_pnl(self, stock_prices, times_to_expiry=None):
        """
        Calculate PnL matrix for given stock prices and times to expiry
        
        Args:
            stock_prices (np.ndarray): 1D or 2D array of stock prices
                If 1D, represents different prices at a single point in time
                If 2D, the shape should be (num_dates, num_prices)
            times_to_expiry (np.ndarray, optional): 1D array of times to expiry in years.
                If None and stock_prices is 2D, assumes the first axis represents
                different times with the last point being expiry (times_to_expiry=0).
                If None and stock_prices is 1D, uses the current time to expiry.
        
        Returns:
            np.ndarray: PnL matrix with same shape as stock_prices
        """
        # Determine input dimensions
        is_1d = len(stock_prices.shape) == 1
        
        if is_1d:
            # Handle 1D input (e.g., checking different prices at a single time)
            if times_to_expiry is None:
                # Use current time to expiry for all prices
                time_to_expiry = self.days_to_expiry / 365.0
                option_values = self.calculate_option_value(stock_prices, time_to_expiry)
            else:
                # If time is provided with 1D prices, broadcast the calculation
                option_values = np.zeros((len(times_to_expiry), len(stock_prices)))
                for i, t in enumerate(times_to_expiry):
                    option_values[i, :] = self.calculate_option_value(stock_prices, t)
                return self._calculate_pnl_from_values(option_values)
        else:
            # Handle 2D input
            num_dates, num_prices = stock_prices.shape
            
            if times_to_expiry is None:
                # Generate times to expiry linearly from current time to expiry
                times_to_expiry = np.linspace(
                    self.days_to_expiry / 365.0, 
                    0,  # Expiry
                    num_dates
                )
            
            # Calculate option values for each combination of date and price
            option_values = np.zeros_like(stock_prices)
            for i, time_to_expiry in enumerate(times_to_expiry):
                option_values[i, :] = self.calculate_option_value(stock_prices[i, :], time_to_expiry)
        
        # Calculate PnL from option values
        return self._calculate_pnl_from_values(option_values)
    
    def _calculate_pnl_from_values(self, option_values):
        """
        Calculate PnL from option values
        
        Args:
            option_values (np.ndarray): Option values
        
        Returns:
            np.ndarray: PnL values with same shape as option_values
        """
        if self.position_type == 'buy':
            # For buying options, PnL = (current value - initial price) * 100 * num_contracts
            return (option_values - self.option_price) * 100 * self.num_contracts
        else:  # sell
            # For selling options, PnL = (initial price - current value) * 100 * num_contracts
            return (self.option_price - option_values) * 100 * self.num_contracts

def calculate_price_probabilities(ticker, current_price, future_dates, price_range, historical_volatility=None):
    """
    Calculate the probability distribution of stock prices at future dates using log-normal distribution
    
    Args:
        ticker (str): Ticker symbol
        current_price (float): Current price of the stock
        future_dates (list): List of future dates to calculate probabilities for
        price_range (np.array): Array of price points to calculate probabilities for
        historical_volatility (float, optional): Historical volatility. If None, will be calculated.
    
    Returns:
        dict: Dictionary with price probabilities and probability matrix
    """
    # Get historical volatility if not provided
    if historical_volatility is None:
        historical_volatility = calculate_historical_volatility(ticker)
    
    # Get current risk-free rate
    risk_free_rate = get_risk_free_rate()
    
    # Get current date
    current_date = datetime.now().date()
    
    # Initialize probability matrix with same dimensions as PnL matrix
    probability_matrix = np.zeros((len(future_dates), len(price_range)))
    
    # Calculate probabilities for each future date
    result = {
        'dates': future_dates,
        'volatility': historical_volatility,
        'risk_free_rate': risk_free_rate,
        'price_range': price_range,
        'probability_matrix': probability_matrix,
        'price_distributions': []
    }
    
    for i, future_date in enumerate(future_dates):
        # Calculate time to future date in years
        days_to_date = (future_date - current_date).days
        if days_to_date < 0:
            continue  # Skip dates in the past
            
        time_to_date_years = days_to_date / 365.0
        
        # Calculate probability density for each price point using log-normal distribution
        probabilities = []
        
        # Log-normal distribution parameters
        # μ = ln(S0) + (r - σ²/2)T
        # σ = σ√T
        mu = math.log(current_price) + (risk_free_rate - 0.5 * historical_volatility**2) * time_to_date_years
        sigma = historical_volatility * math.sqrt(time_to_date_years)
        
        for j, price in enumerate(price_range):
            # Probability density at this price point
            if sigma > 0:
                prob_density = (1 / (price * sigma * math.sqrt(2 * math.pi))) * \
                              math.exp(-((math.log(price) - mu)**2) / (2 * sigma**2))
            else:
                prob_density = 0
                
            probabilities.append(prob_density)
            probability_matrix[i, j] = prob_density
        
        # Normalize probabilities for this date to sum to 1
        if sum(probabilities) > 0:
            normalized_probabilities = [p / sum(probabilities) for p in probabilities]
            # Also normalize the matrix row
            probability_matrix[i, :] = probability_matrix[i, :] / sum(probability_matrix[i, :])
        else:
            normalized_probabilities = [0] * len(probabilities)
        
        result['price_distributions'].append({
            'date': future_date,
            'time_to_date_years': time_to_date_years,
            'probabilities': normalized_probabilities
        })
    
    result['probability_matrix'] = probability_matrix
    return result

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
    
    # Create 2D price grid for the strategy to calculate on
    price_grid = np.zeros((len(date_range), len(price_range)))
    for i in range(len(date_range)):
        price_grid[i, :] = price_range
    
    # Create the option strategy
    strategy = OptionStrategy(
        ticker=ticker,
        option_type=option_type,
        strike_price=strike_price,
        expiry_date=actual_expiry_date,
        current_price=current_price,
        option_price=option_price,
        implied_volatility=implied_volatility,
        position_type=position_type,
        num_contracts=num_contracts,
        risk_free_rate=risk_free_rate
    )
    
    # Calculate option values and PnL matrix using the strategy
    # Create times_to_expiry array (from today to expiry)
    times_to_expiry = np.array([(actual_expiry_date_dt - (start_date_dt + timedelta(days=i))).days / 365.0 
                              for i in range(days_to_expiry + 1)])
    
    pnl_matrix = strategy.calculate_pnl(price_grid, times_to_expiry)
    
    # Calculate price probability distributions using historical volatility (not implied volatility)
    historical_volatility = calculate_historical_volatility(ticker)
    price_probabilities = calculate_price_probabilities(
        ticker=ticker,
        current_price=current_price,
        future_dates=date_range,
        price_range=price_range,
        historical_volatility=historical_volatility
    )
    
    # Calculate option values (needed for some visualizations)
    option_value_matrix = np.zeros((len(date_range), len(price_range)))
    for i, date in enumerate(date_range):
        time_to_expiry = (actual_expiry_date_dt - date).days / 365.0
        for j, price in enumerate(price_range):
            option_value_matrix[i, j] = strategy.calculate_option_value(price, time_to_expiry)
    
    # Return results
    return {
        'ticker': ticker,
        'option_type': option_type,
        'position_type': position_type,
        'strike_price': strike_price,
        'current_price': current_price,
        'option_price': option_price,
        'implied_volatility': implied_volatility,
        'historical_volatility': historical_volatility,
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
        'initial_greeks': strategy.initial_greeks,
        'risk_free_rate': risk_free_rate,
        'price_probabilities': price_probabilities,
        'strategy': strategy  # Include the strategy object in the results
    }

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

def plot_price_probability(price_probabilities, current_price, target_prices=None, save_path=None):
    """
    Plot the probability distribution of stock prices at future dates
    
    Args:
        price_probabilities (dict): Results from calculate_price_probabilities
        current_price (float): Current price of the stock
        target_prices (list, optional): List of target prices to highlight on the plot
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Extract data from price probabilities
    future_dates = price_probabilities['dates']
    volatility = price_probabilities['volatility']
    price_distributions = price_probabilities['price_distributions']
    price_range = price_probabilities['price_range']  # Get the global price range from price_probabilities
    
    # Handle large number of dates - select a subset if there are too many
    max_dates_to_show = 6  # Maximum number of dates to include in the plot
    if len(price_distributions) > max_dates_to_show:
        # Ensure we always include first and last date (start and expiry)
        selected_indices = [0]  # Start with first date
        
        # Add intermediate dates evenly spaced
        if len(price_distributions) > 2:  # Only add intermediate if we have more than 2 dates total
            step = (len(price_distributions) - 1) / (max_dates_to_show - 1)
            selected_indices.extend([int(i * step) for i in range(1, max_dates_to_show - 1)])
        
        # Add the last date (expiry date)
        selected_indices.append(len(price_distributions) - 1)
        
        # Remove any duplicates and sort
        selected_indices = sorted(list(set(selected_indices)))
        
        # Select only these distributions
        selected_distributions = [price_distributions[i] for i in selected_indices]
    else:
        # Use all distributions if we have few enough
        selected_distributions = price_distributions
    
    # Set colors for different dates using a wider color palette to help differentiate
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_distributions)))
    
    # Create figure with a bit more room on the right for the legend
    fig = plt.figure(num="Price Probability Distribution", figsize=(14, 8))
    ax = fig.add_subplot(111)
    
    # Track max probability for y-axis scaling
    max_probability = 0
    
    # Plot probability distributions for selected dates
    line_handles = []  # Store line handles for legend
    for i, dist in enumerate(selected_distributions):
        date = dist['date']
        days_from_now = (date - datetime.now().date()).days
        probabilities = dist['probabilities']
        
        # Update max probability
        if probabilities and max(probabilities) > max_probability:
            max_probability = max(probabilities)
        
        # Plot this date's distribution
        line, = ax.plot(
            price_range,
            probabilities, 
            color=colors[i], 
            label=f"{date.strftime('%Y-%m-%d')} ({days_from_now} days)"
        )
        line_handles.append(line)
    
    # Calculate reasonable x-axis limits to focus on the relevant price range
    # Typically within ±3 standard deviations of current price
    time_to_expiry = (datetime.strptime(price_probabilities['dates'][-1].strftime('%Y-%m-%d'), '%Y-%m-%d').date() - 
                      datetime.now().date()).days / 365.0
    std_dev = current_price * volatility * np.sqrt(time_to_expiry)
    
    # Set x-axis limits to focus on the relevant price range (±3 std dev, but don't go below 0)
    x_min = max(current_price - 3 * std_dev, price_range[0])
    x_max = min(current_price + 3 * std_dev, price_range[-1])
    
    # Add some padding to y-axis (20% above max probability)
    y_max = max_probability * 1.2
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    
    # Add vertical line for current price
    current_price_line = ax.axvline(x=current_price, color='black', linestyle='-', alpha=0.7, 
                                   label=f'Current Price: ${current_price:.2f}')
    
    # Add all special lines to the legend handles list
    legend_handles = line_handles.copy()
    legend_handles.append(current_price_line)
    
    # Add vertical lines for target prices if provided
    target_price_lines = []
    if target_prices:
        for i, price in enumerate(target_prices):
            line = ax.axvline(x=price, color='red', linestyle='--', alpha=0.5, 
                              label=f'Target: ${price:.2f}')
            target_price_lines.append(line)
            legend_handles.append(line)
    
    # Set title and labels
    ax.set_title(f"Stock Price Probability Distribution (Historical Vol: {volatility:.2%})")
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel('Probability Density')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Create a more compact legend and place it to the right of the plot
    # This prevents the legend from taking up space in the plot area
    lgd = ax.legend(handles=legend_handles, 
                   loc='center left', 
                   bbox_to_anchor=(1.02, 0.5),
                   framealpha=0.9,
                   fontsize=9,  # Smaller font size
                   ncol=1)  # Single column for better readability
    
    # Add text annotation showing probability interpretation
    expiry_date = price_probabilities['dates'][-1]
    ax.text(0.02, 0.96, f"Distribution shows probability of stock price at different dates\n"
                        f"Based on historical volatility of {volatility:.1%}\n"
                        f"Expiration: {expiry_date.strftime('%Y-%m-%d')}", 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Adjust layout to make room for the legend
    fig.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust the right margin to accommodate the legend
    
    # Save or display the plot
    if save_path:
        # Save the figure with the legend included (bbox_extra_artists ensures it's not cut off)
        fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    return fig

def simulate_geometric_brownian_motion(ticker, current_price, days_to_simulate, num_simulations=1, 
                                       historical_volatility=None, risk_free_rate=None, seed=None):
    """
    Simulate stock price paths using Geometric Brownian Motion
    
    Args:
        ticker (str): Ticker symbol
        current_price (float): Current price of the stock
        days_to_simulate (int): Number of days to simulate
        num_simulations (int, optional): Number of simulation paths to generate. Defaults to 1.
        historical_volatility (float, optional): Historical volatility. If None, will be calculated.
        risk_free_rate (float, optional): Risk-free rate. If None, will be fetched.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    
    Returns:
        dict: Simulation results including price paths and dates
    """
    logger.info(f"Starting geometric Brownian motion simulation for {ticker}")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Get historical volatility if not provided
    if historical_volatility is None:
        historical_volatility = calculate_historical_volatility(ticker)
        logger.info(f"Using calculated historical volatility: {historical_volatility:.4f}")
    
    # Get risk-free rate if not provided
    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate()
        logger.info(f"Using current risk-free rate: {risk_free_rate:.4f}")
    
    # Generate date range from today to specified days in the future
    start_date = datetime.now().date()
    date_range = [start_date + timedelta(days=i) for i in range(days_to_simulate + 1)]
    
    # Create a time grid (convert days to years for financial calculations)
    dt = 1/252  # Approximately 252 trading days in a year
    num_steps = days_to_simulate
    
    # For simulating business days only (excluding weekends)
    is_business_day = np.array([d.weekday() < 5 for d in date_range])
    business_days = np.where(is_business_day)[0]
    
    # Initialize price paths array
    price_paths = np.zeros((num_simulations, len(date_range)))
    price_paths[:, 0] = current_price  # Set initial price
    
    # Initialize arrays for the efficient simulation
    drift = (risk_free_rate - 0.5 * historical_volatility**2) * dt
    volatility = historical_volatility * np.sqrt(dt)
    
    # Simulate price paths
    for i in range(num_simulations):
        for step in range(1, len(date_range)):
            # Only update on business days
            if date_range[step].weekday() < 5:  # Monday to Friday
                # Generate random shock from normal distribution
                rand_shock = np.random.normal(0, 1)
                
                # Update price using discretized GBM equation
                price_paths[i, step] = price_paths[i, step-1] * np.exp(drift + volatility * rand_shock)
            else:
                # For weekends, carry forward the previous price
                price_paths[i, step] = price_paths[i, step-1]
    
    # Calculate statistics across simulations
    mean_path = np.mean(price_paths, axis=0)
    std_dev = np.std(price_paths, axis=0)
    percentile_5 = np.percentile(price_paths, 5, axis=0)
    percentile_95 = np.percentile(price_paths, 95, axis=0)
    
    # Return simulation results
    return {
        'ticker': ticker,
        'current_price': current_price,
        'historical_volatility': historical_volatility,
        'risk_free_rate': risk_free_rate,
        'date_range': date_range,
        'price_paths': price_paths,
        'mean_path': mean_path,
        'std_dev': std_dev,
        'percentile_5': percentile_5,
        'percentile_95': percentile_95
    }

def plot_price_simulations(simulation_results, num_paths_to_plot=5, save_path=None):
    """
    Plot the simulated price paths
    
    Args:
        simulation_results (dict): Results from simulate_geometric_brownian_motion
        num_paths_to_plot (int, optional): Number of individual paths to plot. Defaults to 5.
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
    
    # Create figure
    fig = plt.figure(num="Stock Price Simulation", figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Plot a subset of individual paths with low opacity
    num_simulations = price_paths.shape[0]
    indices_to_plot = np.random.choice(num_simulations, min(num_paths_to_plot, num_simulations), replace=False)
    
    for i in indices_to_plot:
        ax.plot(date_range, price_paths[i, :], alpha=0.3, linewidth=0.8)
    
    # Plot mean path
    ax.plot(date_range, mean_path, 'b-', linewidth=2, label='Mean Path')
    
    # Plot confidence interval (5th to 95th percentile)
    ax.fill_between(date_range, percentile_5, percentile_95, color='b', alpha=0.2, label='90% Confidence Interval')
    
    # Plot current price point
    ax.plot(date_range[0], current_price, 'ro', markersize=8, label=f'Current Price: ${current_price:.2f}')
    
    # Add horizontal line at current price
    ax.axhline(y=current_price, color='r', linestyle='--', alpha=0.5)
    
    # Set title and labels
    ax.set_title(f"{ticker} Stock Price Simulation\nGeometric Brownian Motion (Historical Vol: {historical_volatility:.2%})")
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price ($)')
    
    # Format x-axis to show dates nicely
    num_dates = min(10, len(date_range))
    date_indices = np.linspace(0, len(date_range)-1, num_dates, dtype=int)
    ax.set_xticks([date_range[i] for i in date_indices])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add annotations explaining the simulation
    textstr = '\n'.join([
        f'Simulated using Geometric Brownian Motion',
        f'Total paths: {price_paths.shape[0]}',
        f'Historical volatility: {historical_volatility:.2%}',
        f'Risk-free rate: {simulation_results["risk_free_rate"]:.2%}'
    ])
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    fig.tight_layout()
    
    # Save or display the plot
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
    
    return fig

if __name__ == "__main__":
    # Calculate a date 6 months in the future for the example
    future_date = (datetime.now() + timedelta(days=180)).strftime('%Y-%m-%d')
    
    try:
        # Example usage
        results = simulate_option_pnl(
            ticker="AAPL",
            option_type="call",
            expiry_date=future_date,  # Use a date 6 months in the future
            target_delta=0.5,
            position_type="buy",
            num_contracts=1
        )
        
        # Print report
        print(generate_simulation_report(results))
        
        # Plot results
        plot_pnl_slices(results)
        
        # Plot price probability distribution as line graph
        plot_price_probability(results['price_probabilities'], results['current_price'], [results['strike_price']])
        
        # Run a GBM simulation and plot the results
        print("\nRunning Geometric Brownian Motion simulation...")
        gbm_results = simulate_geometric_brownian_motion(
            ticker="AAPL",
            current_price=results['current_price'],
            days_to_simulate=results['days_to_expiry'],
            num_simulations=100,  # Generate 100 different price paths
            historical_volatility=results['historical_volatility'],
            risk_free_rate=results['risk_free_rate']
        )
        
        # Plot the simulation results
        plot_price_simulations(gbm_results, num_paths_to_plot=10)
        
    except Exception as e:
        print(f"Error in option simulation: {str(e)}")
        print("Running standalone GBM simulation instead...")
        
        # Get current price of AAPL
        ticker = "AAPL"
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        print(f"Current price of {ticker}: ${current_price:.2f}")
        
        # Run standalone GBM simulation
        gbm_results = simulate_geometric_brownian_motion(
            ticker=ticker,
            current_price=current_price,
            days_to_simulate=180,  # 6 months
            num_simulations=100,
            seed=42  # For reproducibility
        )
        
        # Plot the simulation results
        plot_price_simulations(gbm_results, num_paths_to_plot=10)
    
    # Show all plots
    plt.show() 