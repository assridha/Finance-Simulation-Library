"""
Service for handling option chain data.
"""
from typing import Dict, Any, List, Optional
import yfinance as yf
from datetime import datetime
import pandas as pd

class OptionService:
    """Service for handling option chain data."""
    
    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> Dict[str, Any]:
        """Get option chain data for a symbol.
        
        Args:
            symbol: Stock symbol to get options for
            expiry: Optional expiry date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing option chain data
            
        Raises:
            ValueError: If symbol is invalid or data cannot be fetched
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get all expiry dates if none specified
            expiry_dates = ticker.options
            if not expiry_dates:
                raise ValueError(f"No options data found for symbol: {symbol}")
            
            # Convert expiry dates to YYYY-MM-DD format
            expiry_dates = [datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d') for date in expiry_dates]
            
            # If expiry is specified, validate it
            if expiry:
                # Convert expiry to YYYY-MM-DD format
                try:
                    expiry = datetime.strptime(expiry, '%Y-%m-%d').strftime('%Y-%m-%d')
                except ValueError:
                    raise ValueError(f"Invalid expiry date format: {expiry}. Use YYYY-MM-DD format.")
                
                if expiry not in expiry_dates:
                    raise ValueError(f"Invalid expiry date: {expiry}")
            
            # Get option chain for specified expiry or first available
            target_expiry = expiry or expiry_dates[0]
            chain = ticker.option_chain(target_expiry)
            
            # Format call options
            calls = []
            for _, row in chain.calls.iterrows():
                calls.append({
                    'strike': float(row['strike']),
                    'expiry': target_expiry,
                    'bid': float(row['bid']),
                    'ask': float(row['ask']),
                    'volume': int(row['volume']) if not pd.isna(row['volume']) else 0,
                    'open_interest': int(row['openInterest']) if not pd.isna(row['openInterest']) else 0,
                    'implied_volatility': float(row['impliedVolatility'])
                })
            
            # Format put options
            puts = []
            for _, row in chain.puts.iterrows():
                puts.append({
                    'strike': float(row['strike']),
                    'expiry': target_expiry,
                    'bid': float(row['bid']),
                    'ask': float(row['ask']),
                    'volume': int(row['volume']) if not pd.isna(row['volume']) else 0,
                    'open_interest': int(row['openInterest']) if not pd.isna(row['openInterest']) else 0,
                    'implied_volatility': float(row['impliedVolatility'])
                })
            
            return {
                'symbol': symbol,
                'expiry_dates': expiry_dates,
                'calls': calls,
                'puts': puts,
                'underlying_price': float(ticker.history(period='1d')['Close'].iloc[-1]),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise ValueError(f"Error fetching option chain: {str(e)}") 