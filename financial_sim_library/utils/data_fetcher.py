"""
Data Fetcher Utilities

This module provides functions for fetching financial data from various sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Set up logger
logger = logging.getLogger(__name__)

def fetch_historical_prices(ticker, period="1y", interval="1d", proxy=None):
    """
    Fetch historical price data for a ticker.
    
    Args:
        ticker (str): The ticker symbol
        period (str, optional): The period to fetch data for. Defaults to "1y".
            Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval (str, optional): The interval between data points. Defaults to "1d".
            Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        proxy (str, optional): Proxy URL. Defaults to None.
    
    Returns:
        pd.DataFrame: Historical price data with columns:
            - Open
            - High
            - Low
            - Close
            - Volume
            - Dividends
            - Stock Splits
    """
    try:
        logger.info(f"Fetching historical data for {ticker}, period={period}, interval={interval}")
        stock = yf.Ticker(ticker)
        history = stock.history(period=period, interval=interval, proxy=proxy)
        
        if history.empty:
            logger.warning(f"No historical data available for {ticker}")
            return pd.DataFrame()
        
        logger.info(f"Successfully fetched {len(history)} data points for {ticker}")
        return history
    
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
        return pd.DataFrame()

def fetch_current_price(ticker, proxy=None):
    """
    Fetch the current price of a ticker.
    
    Args:
        ticker (str): The ticker symbol
        proxy (str, optional): Proxy URL. Defaults to None.
    
    Returns:
        float: Current price of the ticker
    """
    try:
        logger.info(f"Fetching current price for {ticker}")
        stock = yf.Ticker(ticker)
        
        # Try using fast_info (new in yfinance)
        try:
            current_price = stock.fast_info['lastPrice']
            logger.info(f"Current price for {ticker}: {current_price}")
            return current_price
        except (KeyError, AttributeError):
            pass
        
        # Fallback to history
        latest_data = stock.history(period="1d")
        
        if latest_data.empty:
            logger.warning(f"No price data available for {ticker}")
            return None
        
        current_price = latest_data['Close'].iloc[-1]
        logger.info(f"Current price for {ticker}: {current_price}")
        return current_price
    
    except Exception as e:
        logger.error(f"Error fetching current price for {ticker}: {str(e)}")
        return None

def fetch_option_chain(ticker, date=None, proxy=None):
    """
    Fetch option chain data for a ticker.
    
    Args:
        ticker (str): The ticker symbol
        date (str, optional): The expiry date in 'YYYY-MM-DD' format. 
            If None, all available expiry dates will be fetched.
        proxy (str, optional): Proxy URL. Defaults to None.
    
    Returns:
        dict: If date is specified, returns a dict with 'calls' and 'puts' DataFrames.
              If date is None, returns a dict of expiry dates mapping to option chain dicts.
    """
    try:
        logger.info(f"Fetching option chain for {ticker}" + 
                   (f", expiry date: {date}" if date else ""))
        
        stock = yf.Ticker(ticker)
        
        # If no specific date is provided, fetch all available option chains
        if date is None:
            # Get all expiry dates
            try:
                expiry_dates = stock.options
                
                if not expiry_dates:
                    logger.warning(f"No option expiry dates available for {ticker}")
                    return {}
                
                logger.info(f"Available expiry dates for {ticker}: {expiry_dates}")
                
                # Fetch option chains for all expiry dates
                all_chains = {}
                for exp_date in expiry_dates:
                    chain = stock.option_chain(exp_date)
                    all_chains[exp_date] = {
                        'calls': chain.calls,
                        'puts': chain.puts
                    }
                
                return all_chains
            
            except Exception as e:
                logger.error(f"Error fetching option expiry dates for {ticker}: {str(e)}")
                return {}
        
        # Fetch option chain for the specified date
        try:
            chain = stock.option_chain(date)
            return {
                'calls': chain.calls,
                'puts': chain.puts
            }
        
        except Exception as e:
            logger.error(f"Error fetching option chain for {ticker}, date {date}: {str(e)}")
            return {}
    
    except Exception as e:
        logger.error(f"Error fetching option data for {ticker}: {str(e)}")
        return {}

def find_closest_expiry_date(target_date, available_dates):
    """
    Find the closest expiry date to the target date.
    
    Args:
        target_date (str): Target expiry date in 'YYYY-MM-DD' format
        available_dates (list): List of available expiry dates in 'YYYY-MM-DD' format
    
    Returns:
        str: The closest available expiry date
    """
    try:
        # Convert dates to datetime objects
        target_date_dt = datetime.strptime(target_date, '%Y-%m-%d')
        available_dates_dt = [datetime.strptime(date, '%Y-%m-%d') for date in available_dates]
        
        # Calculate the difference in days
        days_diff = [(date - target_date_dt).days for date in available_dates_dt]
        
        # Find the date with the smallest absolute difference
        min_diff_idx = np.argmin(np.abs(days_diff))
        closest_date = available_dates[min_diff_idx]
        
        actual_diff = days_diff[min_diff_idx]
        logger.info(f"Closest expiry date to {target_date} is {closest_date} ({actual_diff} days difference)")
        
        return closest_date
    
    except Exception as e:
        logger.error(f"Error finding closest expiry date: {str(e)}")
        if available_dates:
            logger.warning(f"Returning first available expiry date: {available_dates[0]}")
            return available_dates[0]
        else:
            raise ValueError("No available expiry dates provided")

def find_option_by_delta(options_df, target_delta, current_price, option_type='call'):
    """
    Find the option with the closest delta to the target delta.
    
    Args:
        options_df (pd.DataFrame): DataFrame of options (from option_chain)
        target_delta (float): Target delta value (absolute value)
        current_price (float): Current price of the underlying asset
        option_type (str, optional): 'call' or 'put'. Defaults to 'call'.
    
    Returns:
        pd.Series: Option data for the closest match
    """
    try:
        # Ensure target_delta is positive for comparison
        target_delta = abs(target_delta)
        
        # If delta is available in the DataFrame, use it directly
        if 'delta' in options_df.columns:
            # For puts, use absolute value of delta for comparison
            if option_type.lower() == 'put':
                delta_diff = np.abs(np.abs(options_df['delta']) - target_delta)
            else:
                delta_diff = np.abs(options_df['delta'] - target_delta)
                
            closest_idx = delta_diff.argmin()
            return options_df.iloc[closest_idx]
        
        # If delta is not available, estimate it based on strike and current price
        # This is a very rough approximation and works best for ATM options
        logger.warning("Delta not found in option chain, using approximate method")
        
        # Calculate moneyness (S/K for calls, K/S for puts)
        if option_type.lower() == 'call':
            moneyness = current_price / options_df['strike']
            # Rough approximation for call delta: closer to 1 when deeper ITM, closer to 0 when deeper OTM
            approximate_delta = np.clip((moneyness - 0.8) * 2, 0, 1)
        else:  # put
            moneyness = options_df['strike'] / current_price
            # Rough approximation for put delta: closer to -1 when deeper ITM, closer to 0 when deeper OTM
            approximate_delta = np.clip((moneyness - 0.8) * 2, 0, 1)
        
        delta_diff = np.abs(approximate_delta - target_delta)
        closest_idx = delta_diff.argmin()
        
        option = options_df.iloc[closest_idx].copy()
        logger.info(f"Found option with strike {option['strike']}, approximated delta: {approximate_delta[closest_idx]:.4f}")
        
        return option
    
    except Exception as e:
        logger.error(f"Error finding option by delta: {str(e)}")
        if not options_df.empty:
            # If all else fails, return the option closest to ATM
            atm_idx = np.abs(options_df['strike'] - current_price).argmin()
            logger.warning(f"Returning ATM option with strike: {options_df.iloc[atm_idx]['strike']}")
            return options_df.iloc[atm_idx]
        else:
            raise ValueError("Empty options DataFrame provided")

def save_data_to_csv(data, filename, directory="data"):
    """
    Save data to a CSV file.
    
    Args:
        data (pd.DataFrame): The data to save
        filename (str): Name of the file (without extension)
        directory (str, optional): Directory to save the file in. Defaults to "data".
    
    Returns:
        str: Path to the saved file
    """
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(directory, f"{filename}_{timestamp}.csv")
        
        # Save data
        data.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")
        
        return filepath
    
    except Exception as e:
        logger.error(f"Error saving data to CSV: {str(e)}")
        return None 