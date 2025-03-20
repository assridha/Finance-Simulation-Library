"""
Financial Calculations Utilities

This module provides utility functions for various financial calculations.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging

# Set up logger
logger = logging.getLogger(__name__)

def get_risk_free_rate():
    """
    Get the current risk-free rate based on the 3-month Treasury bill yield.
    
    Returns:
        float: Risk-free rate as a decimal (e.g., 0.03 for 3%)
    """
    try:
        # Fetch ^IRX (13-week Treasury Bill) data
        irx = yf.Ticker("^IRX")
        latest_data = irx.history(period="1d")
        
        if latest_data.empty:
            logger.warning("Could not fetch Treasury yield data, using default rate of 3%")
            return 0.03
        
        # Convert from percentage to decimal
        risk_free_rate = latest_data['Close'].iloc[-1] / 100
        logger.info(f"Current risk-free rate: {risk_free_rate:.4f}")
        return risk_free_rate
    
    except Exception as e:
        logger.error(f"Error fetching risk-free rate: {str(e)}")
        logger.warning("Using default risk-free rate of 3%")
        return 0.03

def calculate_historical_volatility(ticker, lookback_days=30):
    """
    Calculate the historical volatility for a given ticker.
    
    Args:
        ticker (str): The ticker symbol
        lookback_days (int, optional): Number of days to look back. Defaults to 30.
    
    Returns:
        float: Historical volatility as a decimal
    """
    try:
        # Fetch historical data
        stock = yf.Ticker(ticker)
        # Get more days to account for weekends and holidays
        buffer_factor = 1.5
        hist_data = stock.history(period=f"{int(lookback_days * buffer_factor)}d")
        
        if hist_data.empty or len(hist_data) < 5:
            logger.warning(f"Not enough historical data for {ticker}, using default volatility of 30%")
            return 0.30
        
        # Calculate daily returns
        returns = hist_data['Close'].pct_change().dropna()
        
        # Get the most recent lookback_days returns
        if len(returns) > lookback_days:
            returns = returns.iloc[-lookback_days:]
        
        # Calculate annualized volatility
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)  # 252 trading days in a year
        
        logger.info(f"Historical volatility for {ticker} over {len(returns)} days: {annualized_volatility:.4f}")
        return annualized_volatility
    
    except Exception as e:
        logger.error(f"Error calculating historical volatility: {str(e)}")
        logger.warning("Using default historical volatility of 30%")
        return 0.30

def calculate_correlation(ticker1, ticker2, lookback_days=252):
    """
    Calculate the correlation between two tickers.
    
    Args:
        ticker1 (str): First ticker symbol
        ticker2 (str): Second ticker symbol
        lookback_days (int, optional): Number of days to look back. Defaults to 252 (1 year).
    
    Returns:
        float: Correlation coefficient between the two tickers
    """
    try:
        # Fetch historical data
        stock1 = yf.Ticker(ticker1)
        stock2 = yf.Ticker(ticker2)
        
        # Get more days to account for weekends and holidays
        buffer_factor = 1.2
        hist_data1 = stock1.history(period=f"{int(lookback_days * buffer_factor)}d")['Close']
        hist_data2 = stock2.history(period=f"{int(lookback_days * buffer_factor)}d")['Close']
        
        if hist_data1.empty or hist_data2.empty or len(hist_data1) < 10 or len(hist_data2) < 10:
            logger.warning(f"Not enough historical data for {ticker1} or {ticker2}, using default correlation of 0")
            return 0.0
        
        # Align dates and calculate returns
        combined = pd.DataFrame({ticker1: hist_data1, ticker2: hist_data2})
        combined = combined.dropna()
        
        # Calculate daily returns
        returns = combined.pct_change().dropna()
        
        # Get the most recent lookback_days returns
        if len(returns) > lookback_days:
            returns = returns.iloc[-lookback_days:]
        
        # Calculate correlation
        correlation = returns[ticker1].corr(returns[ticker2])
        
        logger.info(f"Correlation between {ticker1} and {ticker2} over {len(returns)} days: {correlation:.4f}")
        return correlation
    
    except Exception as e:
        logger.error(f"Error calculating correlation: {str(e)}")
        logger.warning("Using default correlation of 0")
        return 0.0

def calculate_beta(ticker, benchmark="^GSPC", lookback_days=252):
    """
    Calculate the beta of a stock relative to a benchmark.
    
    Args:
        ticker (str): The ticker symbol
        benchmark (str, optional): Benchmark ticker. Defaults to "^GSPC" (S&P 500).
        lookback_days (int, optional): Number of days to look back. Defaults to 252 (1 year).
    
    Returns:
        float: Beta value
    """
    try:
        # Fetch historical data
        stock = yf.Ticker(ticker)
        market = yf.Ticker(benchmark)
        
        # Get more days to account for weekends and holidays
        buffer_factor = 1.2
        hist_data_stock = stock.history(period=f"{int(lookback_days * buffer_factor)}d")['Close']
        hist_data_market = market.history(period=f"{int(lookback_days * buffer_factor)}d")['Close']
        
        if hist_data_stock.empty or hist_data_market.empty or len(hist_data_stock) < 10 or len(hist_data_market) < 10:
            logger.warning(f"Not enough historical data for {ticker} or {benchmark}, using default beta of 1")
            return 1.0
        
        # Align dates
        combined = pd.DataFrame({'stock': hist_data_stock, 'market': hist_data_market})
        combined = combined.dropna()
        
        # Calculate daily returns
        returns = combined.pct_change().dropna()
        
        # Get the most recent lookback_days returns
        if len(returns) > lookback_days:
            returns = returns.iloc[-lookback_days:]
        
        # Calculate beta (covariance / variance)
        covariance = returns['stock'].cov(returns['market'])
        market_variance = returns['market'].var()
        
        if market_variance == 0:
            logger.warning(f"Market variance is zero, using default beta of 1")
            return 1.0
            
        beta = covariance / market_variance
        
        logger.info(f"Beta for {ticker} relative to {benchmark} over {len(returns)} days: {beta:.4f}")
        return beta
    
    except Exception as e:
        logger.error(f"Error calculating beta: {str(e)}")
        logger.warning("Using default beta of 1")
        return 1.0 