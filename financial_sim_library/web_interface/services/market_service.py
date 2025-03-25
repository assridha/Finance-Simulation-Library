"""
Service for handling market data and price simulations.
"""
from typing import Dict, Any, Optional
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

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
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No historical data found for symbol: {symbol}")
            
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
        """Run a Monte Carlo price simulation.
        
        Args:
            symbol (str): Stock symbol to simulate.
            days_to_simulate (int): Number of days to simulate.
            num_simulations (int): Number of simulation paths.
            
        Returns:
            dict: Simulation results.
            
        Raises:
            ValueError: If the symbol is invalid or simulation fails.
        """
        ticker = self._validate_symbol(symbol)
        
        try:
            # Get historical data for parameter estimation
            data = ticker.history(period='1y')
            if data.empty:
                raise ValueError(f"No data available for symbol: {symbol}")
            
            # Extract closing prices
            prices = np.array(data['Close'])
            returns = np.log(prices[1:] / prices[:-1])
            
            # Calculate parameters
            mu = np.mean(returns)
            sigma = np.std(returns)
            S0 = prices[-1]
            dt = 1/252  # Daily time step
            
            # Run simulations
            paths = np.zeros((days_to_simulate + 1, num_simulations))  # +1 to include initial day
            paths[0] = S0
            
            for t in range(1, days_to_simulate + 1):  # +1 to include initial day
                z = np.random.standard_normal(num_simulations)
                paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            
            # Calculate statistics
            final_prices = paths[-1]
            mean_price = np.mean(final_prices)
            
            return {
                'symbol': symbol,
                'current_price': float(S0),
                'simulated_paths': paths.T.tolist(),  # Transpose to get paths as rows
                'statistics': {
                    'mean': float(mean_price),
                    'std': float(np.std(final_prices)),
                    'min': float(np.min(final_prices)),
                    'max': float(np.max(final_prices)),
                    'expected_return': float((mean_price - S0) / S0),
                    'prob_above_current': float(np.mean(final_prices > S0))
                },
                'timestamps': [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                             for i in range(days_to_simulate + 1)],  # +1 to include initial day
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            raise ValueError(f"Error running simulation for {symbol}: {str(e)}") 