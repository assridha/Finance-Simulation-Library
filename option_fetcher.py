import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import bisect
from typing import Optional, List, Dict, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_option_data(
    ticker: str,
    option_type: str,
    target_expiry_date: str,
    target_delta: float
) -> Dict[str, Union[str, float, pd.DataFrame]]:
    """
    Fetch option data for a given ticker, option type, expiry date, and delta.
    
    Args:
        ticker (str): The ticker symbol (e.g., 'AAPL', 'SPY')
        option_type (str): 'call' or 'put'
        target_expiry_date (str): Target expiry date in 'YYYY-MM-DD' format
        target_delta (float): Target delta value (absolute value, e.g., 0.5)
    
    Returns:
        Dict containing:
            - 'option_data': DataFrame with the selected option data
            - 'expiry_date': The actual expiry date used (may differ from target if exact match not found)
            - 'delta': The actual delta value of the selected option
            - 'strike': The strike price of the selected option
            - 'ticker': The ticker symbol
            - 'option_type': The option type ('call' or 'put')
            - 'underlying_price': Current price of the underlying asset
    """
    # Validate inputs
    if option_type.lower() not in ['call', 'put']:
        raise ValueError("option_type must be either 'call' or 'put'")
    
    try:
        target_date = datetime.strptime(target_expiry_date, '%Y-%m-%d').date()
    except ValueError:
        raise ValueError("target_expiry_date must be in 'YYYY-MM-DD' format")
    
    # Get stock data
    logger.info(f"Fetching data for ticker: {ticker}")
    stock = yf.Ticker(ticker)
    
    # Get available expiry dates
    try:
        expiry_dates = stock.options
        
        if not expiry_dates or len(expiry_dates) == 0:
            raise ValueError(f"No options available for {ticker}")
        
        logger.info(f"Available expiry dates: {len(expiry_dates)} dates")
        
        # Convert expiry dates to datetime objects for comparison
        expiry_dates_dt = [datetime.strptime(date, '%Y-%m-%d').date() for date in expiry_dates]
        
        # Find closest expiry date
        closest_date_idx = min(range(len(expiry_dates_dt)), 
                              key=lambda i: abs((expiry_dates_dt[i] - target_date).days))
        closest_expiry = expiry_dates[closest_date_idx]
        
        logger.info(f"Selected expiry date: {closest_expiry} (closest to target: {target_expiry_date})")
        
        # Get option chain for the closest expiry date
        # In yfinance 0.2.28+, option_chain() also returns underlying data
        option_chain = stock.option_chain(closest_expiry)
        
        # Get underlying price - available in newer versions of yfinance
        try:
            # First try to get from option_chain if available (yfinance 0.2.28+)
            if hasattr(option_chain, 'underlying'):
                underlying_price = option_chain.underlying
                logger.info(f"Got underlying price from option_chain: {underlying_price}")
            else:
                # Fall back to fast_info
                underlying_price = stock.fast_info['lastPrice']
                logger.info(f"Got underlying price from fast_info: {underlying_price}")
        except Exception as e:
            # Last resort: get from history
            logger.warning(f"Error getting underlying price from fast methods: {str(e)}")
            underlying_price = stock.history(period='1d')['Close'].iloc[-1]
            logger.info(f"Got underlying price from history: {underlying_price}")
        
        # Select calls or puts
        if option_type.lower() == 'call':
            options_df = option_chain.calls
        else:  # put
            options_df = option_chain.puts
        
        # Check if we have any options data
        if options_df.empty:
            raise ValueError(f"No {option_type} options available for {ticker} with expiry {closest_expiry}")
        
        logger.info(f"Found {len(options_df)} {option_type} options for expiry {closest_expiry}")
        
        # Handle delta calculation
        # Latest yfinance may include greeks in some cases, but not always
        if 'delta' not in options_df.columns:
            logger.info("Delta not provided in options data, calculating approximation")
            
            # Calculate approximate delta for calls and puts
            if option_type.lower() == 'call':
                # For calls, delta increases as strike decreases
                options_df['delta'] = 0.5 + 0.5 * np.tanh((underlying_price - options_df['strike']) / (underlying_price * 0.1))
            else:
                # For puts, delta decreases (becomes more negative) as strike decreases
                options_df['delta'] = -0.5 - 0.5 * np.tanh((underlying_price - options_df['strike']) / (underlying_price * 0.1))
        
        # For puts, we need to work with absolute delta values for finding the closest
        delta_values = options_df['delta'].abs() if option_type.lower() == 'put' else options_df['delta']
        
        # Find option with closest delta
        closest_delta_idx = (delta_values - abs(target_delta)).abs().idxmin()
        selected_option = options_df.loc[closest_delta_idx]
        
        logger.info(f"Selected option with delta {selected_option['delta']:.4f} (target: {target_delta})")
        logger.info(f"Strike price: {selected_option['strike']}")
        
        # Return the result
        return {
            'option_data': selected_option,
            'expiry_date': closest_expiry,
            'delta': selected_option['delta'],
            'strike': selected_option['strike'],
            'ticker': ticker,
            'option_type': option_type.lower(),
            'underlying_price': underlying_price
        }
    
    except Exception as e:
        logger.error(f"Error in primary method: {str(e)}")
        # Try alternative approach if the standard approach fails
        try:
            logger.info("Trying alternative approach to fetch options data")
            
            # Get current stock price for delta calculation
            try:
                underlying_price = stock.fast_info['lastPrice']
            except Exception:
                underlying_price = stock.history(period='1d')['Close'].iloc[-1]
            
            logger.info(f"Current price of {ticker}: {underlying_price}")
            
            # Try to get options data using the newer API
            all_options_data = {}
            
            # Manually fetch each expiry date's options
            for exp_date in expiry_dates:
                try:
                    chain = stock.option_chain(exp_date)
                    all_options_data[exp_date] = {
                        'calls': chain.calls,
                        'puts': chain.puts,
                        'underlying': underlying_price
                    }
                except Exception as exp_error:
                    logger.warning(f"Could not fetch options for expiry {exp_date}: {str(exp_error)}")
            
            if not all_options_data:
                raise ValueError(f"Could not fetch any options data for {ticker}")
            
            # Find closest expiry date
            available_expiries = list(all_options_data.keys())
            expiry_dates_dt = [datetime.strptime(date, '%Y-%m-%d').date() for date in available_expiries]
            
            closest_date_idx = min(range(len(expiry_dates_dt)), 
                                  key=lambda i: abs((expiry_dates_dt[i] - target_date).days))
            closest_expiry = available_expiries[closest_date_idx]
            
            logger.info(f"Selected expiry date (alternative method): {closest_expiry}")
            
            # Get option chain for the closest expiry
            option_chain = all_options_data[closest_expiry]
            
            # Select calls or puts
            if option_type.lower() == 'call':
                options_df = option_chain['calls']
            else:
                options_df = option_chain['puts']
            
            # Check if we have any options data
            if options_df.empty:
                raise ValueError(f"No {option_type} options available for {ticker} with expiry {closest_expiry}")
            
            # Handle delta calculation
            if 'delta' not in options_df.columns:
                logger.info("Delta not provided in options data, calculating approximation (alternative method)")
                
                # Calculate approximate delta
                if option_type.lower() == 'call':
                    options_df['delta'] = 0.5 + 0.5 * np.tanh((underlying_price - options_df['strike']) / (underlying_price * 0.1))
                else:
                    options_df['delta'] = -0.5 - 0.5 * np.tanh((underlying_price - options_df['strike']) / (underlying_price * 0.1))
            
            # Find closest delta
            delta_values = options_df['delta'].abs() if option_type.lower() == 'put' else options_df['delta']
            closest_delta_idx = (delta_values - abs(target_delta)).abs().idxmin()
            selected_option = options_df.loc[closest_delta_idx]
            
            logger.info(f"Selected option with delta {selected_option['delta']:.4f} (target: {target_delta})")
            logger.info(f"Strike price: {selected_option['strike']}")
            
            return {
                'option_data': selected_option,
                'expiry_date': closest_expiry,
                'delta': selected_option['delta'],
                'strike': selected_option['strike'],
                'ticker': ticker,
                'option_type': option_type.lower(),
                'underlying_price': underlying_price
            }
            
        except Exception as nested_e:
            logger.error(f"Alternative method also failed: {str(nested_e)}")
            raise ValueError(f"Failed to fetch option data for {ticker}: {str(e)}. Additional error: {str(nested_e)}") 