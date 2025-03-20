"""
Geometric Brownian Motion (GBM) Model

This module implements the Geometric Brownian Motion model for stock price simulation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import logging
from .base_model import StockPriceModel
from ...utils.financial_calcs import get_risk_free_rate

# Set up logger
logger = logging.getLogger(__name__)

class GBMModel(StockPriceModel):
    """
    Geometric Brownian Motion model for stock price simulation.
    
    The GBM model is described by the stochastic differential equation:
    dS = μS dt + σS dW
    
    where:
    - S is the stock price
    - μ is the drift (expected return)
    - σ is the volatility
    - dW is a Wiener process (random term)
    """
    
    def __init__(self, ticker, current_price=None, volatility=None, risk_free_rate=None, seed=None):
        """
        Initialize the GBM model.
        
        Args:
            ticker (str): The ticker symbol for the stock
            current_price (float, optional): The current price of the stock.
                If None, will be fetched from data source.
            volatility (float, optional): Historical volatility.
                If None, will be calculated from historical data.
            risk_free_rate (float, optional): Risk-free rate.
                If None, will be fetched from data source.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(ticker, current_price)
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.seed = seed
        
        # Fetch current price if not provided
        if self.current_price is None:
            self._fetch_current_price()
            
        # Additional validation
        if self.volatility is not None and (self.volatility <= 0 or self.volatility > 1):
            raise ValueError("Volatility must be a positive number between 0 and 1")
        
        if self.risk_free_rate is not None and self.risk_free_rate < 0:
            raise ValueError("Risk-free rate cannot be negative")
    
    def _fetch_current_price(self):
        """Fetch the current price of the stock."""
        try:
            stock = yf.Ticker(self.ticker)
            self.current_price = stock.history(period="1d")['Close'].iloc[-1]
            logger.info(f"Fetched current price for {self.ticker}: ${self.current_price:.2f}")
        except Exception as e:
            logger.error(f"Error fetching current price for {self.ticker}: {str(e)}")
            raise ValueError(f"Could not fetch current price for {self.ticker}")
    
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
        # Get historical data if not provided
        if historical_data is None:
            try:
                stock = yf.Ticker(self.ticker)
                # Get slightly more data to ensure we have enough trading days
                buffer_factor = 1.5
                historical_data = stock.history(period=f"{int(lookback_days * buffer_factor)}d")
                
                if historical_data.empty:
                    raise ValueError(f"No historical data available for {self.ticker}")
                    
                logger.info(f"Fetched {len(historical_data)} days of historical data for {self.ticker}")
            except Exception as e:
                logger.error(f"Error fetching historical data for {self.ticker}: {str(e)}")
                raise ValueError(f"Could not fetch historical data for {self.ticker}")
        
        # Calculate historical volatility if not provided
        if self.volatility is None:
            # Calculate daily returns
            returns = historical_data['Close'].pct_change().dropna()
            
            # If we have too few data points, we might need to adjust our approach
            if len(returns) < 5:
                logger.warning(f"Not enough historical data for {self.ticker}, using default volatility of 0.3")
                self.volatility = 0.3
            else:
                # Annualize the standard deviation of daily returns
                self.volatility = returns.std() * np.sqrt(252)
                logger.info(f"Calculated historical volatility for {self.ticker}: {self.volatility:.4f}")
        
        # Get risk-free rate if not provided
        if self.risk_free_rate is None:
            self.risk_free_rate = get_risk_free_rate()
            logger.info(f"Using risk-free rate: {self.risk_free_rate:.4f}")
        
        return {
            'volatility': self.volatility,
            'risk_free_rate': self.risk_free_rate,
            'ticker': self.ticker,
            'current_price': self.current_price
        }
    
    def simulate(self, days_to_simulate, num_simulations, start_date=None):
        """
        Simulate future stock price paths using Geometric Brownian Motion.
        
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
        logger.info(f"Starting GBM simulation for {self.ticker}: {num_simulations} paths over {days_to_simulate} days")
        
        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Ensure model is calibrated
        if self.volatility is None or self.risk_free_rate is None:
            self.calibrate()
        
        # Set start date if not provided
        if start_date is None:
            start_date = datetime.now()
        
        # Generate business dates for simulation
        business_dates = self._generate_business_dates(start_date, days_to_simulate)
        
        # Create a time grid (convert days to years for financial calculations)
        dt = 1/252  # Approximately 252 trading days in a year
        
        # Initialize price paths array
        price_paths = np.zeros((num_simulations, len(business_dates)))
        price_paths[:, 0] = self.current_price  # Set initial price
        
        # Parameters for the simulation
        drift = (self.risk_free_rate - 0.5 * self.volatility**2) * dt
        vol_sqrt_dt = self.volatility * np.sqrt(dt)
        
        # Simulate price paths
        for i in range(num_simulations):
            for t in range(1, len(business_dates)):
                # Generate random shock from standard normal distribution
                rand_shock = np.random.normal(0, 1)
                
                # Update price using discretized GBM equation
                price_paths[i, t] = price_paths[i, t-1] * np.exp(drift + vol_sqrt_dt * rand_shock)
        
        # Create simulation results dictionary
        simulation_results = {
            'ticker': self.ticker,
            'current_price': self.current_price,
            'price_paths': price_paths,
            'dates': business_dates,
            'parameters': {
                'volatility': self.volatility,
                'risk_free_rate': self.risk_free_rate,
                'days_simulated': days_to_simulate,
                'num_simulations': num_simulations
            }
        }
        
        # Add statistical analysis
        simulation_results = self.analyze_results(simulation_results)
        
        return simulation_results 