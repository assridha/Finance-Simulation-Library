"""
Service for handling market data and price simulations.
"""
from typing import Dict, Any, Optional
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from ...stock_simulator.models.gbm import GBMModel

class MarketService:
    """Service for handling market data and price simulations."""
    
    def _validate_symbol(self, symbol):
        """Validate a stock symbol.
        
        Args:
            symbol (str): Stock symbol to validate.
            
        Raises:
            ValueError: If the symbol is invalid.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                raise ValueError(f"Invalid symbol: {symbol}")
            
            return ticker
        except Exception as e:
            raise ValueError(f"Invalid symbol: {symbol}")
    
    def get_market_data(self, symbol):
        """Get current market data for a symbol.
        
        Args:
            symbol (str): Stock symbol to fetch data for.
            
        Returns:
            dict: Market data including price, change, and volatility.
            
        Raises:
            ValueError: If the symbol is invalid or data cannot be fetched.
        """
        ticker = self._validate_symbol(symbol)
        
        try:
            info = ticker.info
            
            # Get historical data for volatility calculation
            hist = ticker.history(period='1mo')
            if len(hist) < 2:
                raise ValueError(f"Insufficient data for symbol: {symbol}")
            
            # Calculate daily returns and volatility
            returns = np.log(hist['Close'] / hist['Close'].shift(1))
            volatility = np.std(returns.dropna()) * np.sqrt(252)  # Annualized volatility
            
            return {
                'symbol': symbol,
                'price': info['regularMarketPrice'],
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('regularMarketVolume'),
                'volatility': float(volatility),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            raise ValueError(f"Error fetching market data for {symbol}: {str(e)}")
    
    def get_historical_data(self, symbol, period='1y', interval='1d'):
        """Get historical price data for a symbol.
        
        Args:
            symbol (str): Stock symbol to fetch data for.
            period (str): Time period to fetch data for (e.g., '1d', '1mo', '1y').
            interval (str): Data interval (e.g., '1m', '1h', '1d').
            
        Returns:
            dict: Historical price data.
            
        Raises:
            ValueError: If the symbol is invalid or data cannot be fetched.
        """
        ticker = self._validate_symbol(symbol)
        
        try:
            # Fetch slightly more data to ensure we have enough after filtering
            fetch_period = '70d' if period == '2mo' else period
            data = ticker.history(period=fetch_period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No historical data found for symbol: {symbol}")
            
            # If period is 2mo, filter to exactly last 60 days
            if period == '2mo':
                data = data.last('60D')
            
            return {
                'timestamps': data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'open': data['Open'].tolist(),
                'high': data['High'].tolist(),
                'low': data['Low'].tolist(),
                'close': data['Close'].tolist(),
                'volume': data['Volume'].tolist(),
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            raise ValueError(f"Error fetching historical data for {symbol}: {str(e)}")
    
    def run_price_simulation(self, symbol, days_to_simulate=252, num_simulations=1000):
        """Run a Monte Carlo price simulation using the GBMModel.
        
        Args:
            symbol (str): Stock symbol to simulate.
            days_to_simulate (int): Number of days to simulate.
            num_simulations (int): Number of simulation paths.
            
        Returns:
            dict: Simulation results.
            
        Raises:
            ValueError: If the symbol is invalid or simulation fails.
        """
        try:
            # Create and calibrate the GBM model
            model = GBMModel(
                ticker=symbol,
                days_to_simulate=days_to_simulate,
                num_simulations=num_simulations
            )
            
            # Run simulation
            results = model.simulate()
            
            # Format results to match the expected API response
            return {
                'symbol': symbol,
                'current_price': float(results['current_price']),
                'simulated_paths': results['price_paths'].tolist(),
                'statistics': {
                    'mean': float(results['statistics']['mean']),
                    'std': float(results['statistics']['std']),
                    'min': float(results['statistics']['min']),
                    'max': float(results['statistics']['max']),
                    'expected_return': float(results['statistics']['expected_return'] / 100),  # Convert from percentage
                    'prob_above_current': float(results['statistics']['prob_above_current'] / 100),  # Convert from percentage
                    'volatility': float(results['parameters']['volatility'] * 100)  # Convert to percentage
                },
                'timestamps': [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                             for i in range(days_to_simulate + 1)],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            raise ValueError(f"Error running simulation for {symbol}: {str(e)}") 