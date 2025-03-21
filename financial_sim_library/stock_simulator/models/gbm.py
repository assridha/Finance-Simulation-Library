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
from typing import Dict, Any, Optional, List
from .growth_models import GrowthModel, FixedGrowthModel

# Set up logger
logger = logging.getLogger(__name__)

class GBMModel(StockPriceModel):
    """Geometric Brownian Motion model with optional growth component."""
    
    def __init__(
        self,
        ticker: str,
        growth_model: Optional[GrowthModel] = None,
        risk_free_rate: Optional[float] = None,
        volatility: Optional[float] = None,
        days_to_simulate: int = 30,
        num_simulations: int = 1000
    ):
        """
        Initialize the GBM model.
        
        Args:
            ticker: Stock ticker symbol
            growth_model: Optional growth model to incorporate into the simulation
            risk_free_rate: Optional risk-free rate (if None, will be fetched)
            volatility: Optional volatility (if None, will be calculated from historical data)
            days_to_simulate: Number of days to simulate
            num_simulations: Number of simulation paths to generate
        """
        super().__init__(ticker)
        self.growth_model = growth_model or FixedGrowthModel(growth_rate=0.0)
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.days_to_simulate = days_to_simulate
        self.num_simulations = num_simulations
        self.current_price = None
        self.simulation_results = None
        
        # Calibrate the model on initialization
        self.calibrate()
    
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
        
        # Get current price
        self.current_price = historical_data['Close'].iloc[-1]
        logger.info(f"Current price for {self.ticker}: ${self.current_price:.2f}")
        
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
    
    def simulate(self, **kwargs) -> Dict[str, Any]:
        """
        Run the GBM simulation with growth component.
        
        Args:
            **kwargs: Additional parameters including:
                - days_to_simulate: Number of days to simulate (default: self.days_to_simulate)
                - num_simulations: Number of simulation paths (default: self.num_simulations)
                - Additional parameters to pass to the growth model
            
        Returns:
            Dict containing simulation results and statistics
        """
        if self.current_price is None:
            self.calibrate()
        
        # Get simulation parameters from kwargs or use instance defaults
        days_to_simulate = kwargs.get('days_to_simulate', self.days_to_simulate)
        num_simulations = kwargs.get('num_simulations', self.num_simulations)
        
        # Time parameters
        dt = 1/252  # Daily time step
        t = np.arange(0, days_to_simulate * dt, dt)
        
        # Initialize price paths
        price_paths = np.zeros((num_simulations, len(t)))
        price_paths[:, 0] = self.current_price
        
        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), (num_simulations, len(t)-1))
        
        # Simulate price paths
        for i in range(1, len(t)):
            # Calculate growth rate for current time step
            growth_rate = self.growth_model.calculate_growth_rate(t[i-1], dt, **kwargs)
            
            # GBM formula with growth component
            drift = growth_rate + (self.risk_free_rate - 0.5 * self.volatility**2) * dt
            diffusion = self.volatility * dW[:, i-1]
            
            # Update prices
            price_paths[:, i] = price_paths[:, i-1] * np.exp(drift + diffusion)
        
        # Calculate statistics
        final_prices = price_paths[:, -1]
        returns = (final_prices - self.current_price) / self.current_price * 100
        
        stats = {
            'mean': np.mean(final_prices),
            'median': np.median(final_prices),
            'std': np.std(final_prices),
            'min': np.min(final_prices),
            'max': np.max(final_prices),
            'expected_return': np.mean(returns),
            'return_std': np.std(returns),
            'prob_above_current': np.mean(final_prices > self.current_price) * 100,
            'prob_above_10_percent': np.mean(returns > 10) * 100,
            'prob_below_10_percent': np.mean(returns < -10) * 100
        }
        
        # Store results
        self.simulation_results = {
            'price_paths': price_paths,
            'time_points': t,
            'statistics': stats,
            'current_price': self.current_price,
            'parameters': {
                'volatility': self.volatility,
                'risk_free_rate': self.risk_free_rate,
                'growth_model': self.growth_model.get_parameters()
            }
        }
        
        return self.simulation_results
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current model parameters."""
        return {
            'ticker': self.ticker,
            'current_price': self.current_price,
            'volatility': self.volatility,
            'risk_free_rate': self.risk_free_rate,
            'growth_model': self.growth_model.get_parameters() if self.growth_model else None,
            'days_to_simulate': self.days_to_simulate,
            'num_simulations': self.num_simulations
        }
    
    def set_parameters(self, **kwargs) -> None:
        """Set model parameters."""
        if 'ticker' in kwargs:
            self.ticker = kwargs['ticker']
        if 'volatility' in kwargs:
            self.volatility = kwargs['volatility']
        if 'risk_free_rate' in kwargs:
            self.risk_free_rate = kwargs['risk_free_rate']
        if 'growth_model' in kwargs:
            self.growth_model = kwargs['growth_model']
        if 'days_to_simulate' in kwargs:
            self.days_to_simulate = kwargs['days_to_simulate']
        if 'num_simulations' in kwargs:
            self.num_simulations = kwargs['num_simulations']
    
    def _fetch_current_price(self):
        """Fetch the current price of the stock."""
        try:
            stock = yf.Ticker(self.ticker)
            self.current_price = stock.history(period="1d")['Close'].iloc[-1]
            logger.info(f"Fetched current price for {self.ticker}: ${self.current_price:.2f}")
        except Exception as e:
            logger.error(f"Error fetching current price for {self.ticker}: {str(e)}")
            raise ValueError(f"Could not fetch current price for {self.ticker}") 