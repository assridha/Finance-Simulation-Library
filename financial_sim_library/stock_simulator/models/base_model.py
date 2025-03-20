"""
Base Model for Stock Price Simulations

This module defines the abstract base class for all stock price simulation models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logger
logger = logging.getLogger(__name__)

class StockPriceModel(ABC):
    """
    Abstract base class for all stock price simulation models.
    
    All stock price simulation models should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, ticker, current_price=None):
        """
        Initialize the stock price model.
        
        Args:
            ticker (str): The ticker symbol for the stock
            current_price (float, optional): The current price of the stock.
                If None, will be fetched from data source.
        """
        self.ticker = ticker
        self.current_price = current_price
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate the inputs provided to the model."""
        if not isinstance(self.ticker, str):
            raise TypeError("ticker must be a string")
        
        if self.current_price is not None and not isinstance(self.current_price, (int, float)):
            raise TypeError("current_price must be a number")
            
        if self.current_price is not None and self.current_price <= 0:
            raise ValueError("current_price must be positive")
    
    @abstractmethod
    def calibrate(self, historical_data=None, lookback_days=30):
        """
        Calibrate the model parameters based on historical data.
        
        Args:
            historical_data (pd.DataFrame, optional): Historical price data.
                If None, will be fetched using the ticker.
            lookback_days (int, optional): Number of days to look back for 
                calibration if historical_data is not provided.
                
        Returns:
            dict: Dictionary of calibrated model parameters
        """
        pass
    
    @abstractmethod
    def simulate(self, days_to_simulate, num_simulations, start_date=None):
        """
        Simulate future stock price paths.
        
        Args:
            days_to_simulate (int): Number of days to simulate
            num_simulations (int): Number of simulation paths to generate
            start_date (datetime, optional): Start date for the simulation.
                If None, uses current date.
                
        Returns:
            dict: Simulation results containing:
                - price_paths: 2D array of price paths
                - dates: Array of dates for the simulation
                - parameters: Model parameters used
                - statistics: Statistical summary of the simulation
        """
        pass
    
    def _generate_business_dates(self, start_date, days_to_simulate):
        """
        Generate a sequence of business dates for the simulation.
        
        Args:
            start_date (datetime): Start date for the sequence
            days_to_simulate (int): Number of business days to generate
            
        Returns:
            pd.DatetimeIndex: Sequence of business dates
        """
        if start_date is None:
            start_date = datetime.now()
            
        # Generate a range of dates with a buffer for weekends and holidays
        all_dates = pd.date_range(
            start=start_date, 
            periods=days_to_simulate * 2,  # Buffer for weekends/holidays
            freq='D'
        )
        
        # Filter to business days only
        business_dates = pd.bdate_range(
            start=start_date,
            periods=days_to_simulate
        )
        
        return business_dates
    
    def analyze_results(self, simulation_results):
        """
        Analyze the simulation results to extract statistical insights.
        
        Args:
            simulation_results (dict): Simulation results from the simulate method
            
        Returns:
            dict: Enhanced results with statistical analysis
        """
        price_paths = simulation_results['price_paths']
        
        # Final price distribution
        final_prices = price_paths[:, -1]
        
        stats = {
            'mean': np.mean(final_prices),
            'median': np.median(final_prices),
            'std': np.std(final_prices),
            'min': np.min(final_prices),
            'max': np.max(final_prices),
            'percentiles': {
                '10': np.percentile(final_prices, 10),
                '25': np.percentile(final_prices, 25),
                '50': np.percentile(final_prices, 50),
                '75': np.percentile(final_prices, 75),
                '90': np.percentile(final_prices, 90)
            }
        }
        
        current_price = simulation_results['current_price']
        stats['expected_return'] = (stats['mean'] / current_price - 1) * 100
        stats['prob_above_current'] = np.mean(final_prices > current_price) * 100
        stats['prob_10pct_up'] = np.mean(final_prices > current_price * 1.1) * 100
        stats['prob_10pct_down'] = np.mean(final_prices < current_price * 0.9) * 100
        
        # Add stats to results
        simulation_results['statistics'] = stats
        
        return simulation_results 