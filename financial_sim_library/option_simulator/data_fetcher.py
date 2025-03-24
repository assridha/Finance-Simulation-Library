import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from .strategies.base import OptionContract
import numpy as np
from ..stock_simulator.models.gbm import GBMModel
from ..utils.financial_calcs import get_risk_free_rate
from ..utils.data_fetcher import (
    fetch_current_price,
    fetch_option_chain,
    find_closest_expiry_date,
    find_option_by_delta
)

class MarketDataFetcher:
    """Fetcher for live market data using yfinance."""
    
    def __init__(self):
        self.cache = {}
        self.option_chain_cache = {}
        self.stock_price_cache = {}
        self.volatility_cache = {}
        self.risk_free_rate_value = None
    
    def get_stock_price(self, symbol: str) -> float:
        """Get current stock price."""
        if symbol in self.stock_price_cache:
            return self.stock_price_cache[symbol]
        
        price = fetch_current_price(symbol)
        self.stock_price_cache[symbol] = price
        return price
    
    def get_option_chain(self, 
                        symbol: str, 
                        expiration_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get option chain for a symbol."""
        # Convert datetime to string format if provided
        exp_date_str = expiration_date.strftime('%Y-%m-%d') if expiration_date else None
        
        # Check cache first
        cache_key = f"{symbol}_{exp_date_str}"
        if cache_key in self.option_chain_cache:
            return self.option_chain_cache[cache_key]
        
        # Get option chain using utility function
        chain_data = fetch_option_chain(symbol)
        
        if not chain_data:
            return pd.DataFrame()
        
        # Get available dates
        available_dates = list(chain_data.keys())
        if not available_dates:
            return pd.DataFrame()
        
        # If no specific date was requested, get the first available date
        if expiration_date is None:
            exp_date_str = available_dates[0]
        else:
            # If the requested date isn't available, find the closest available date
            exp_date_str = find_closest_expiry_date(exp_date_str, available_dates)
        
        chain_data = chain_data[exp_date_str]
        
        # Combine calls and puts
        calls = chain_data['calls']
        puts = chain_data['puts']
        
        # Add option type column
        calls['option_type'] = 'call'
        puts['option_type'] = 'put'
        
        # Add expiration date column
        calls['expiration'] = exp_date_str
        puts['expiration'] = exp_date_str
        
        # Get current stock price and add it to each option
        current_price = self.get_stock_price(symbol)
        calls['underlyingPrice'] = current_price
        puts['underlyingPrice'] = current_price
        
        # Combine and sort by strike price
        chain = pd.concat([calls, puts])
        chain = chain.sort_values('strike')
        
        # Cache the result
        self.option_chain_cache[cache_key] = chain
        
        return chain
    
    def get_risk_free_rate(self) -> float:
        """Get current risk-free rate."""
        if self.risk_free_rate_value is None:
            self.risk_free_rate_value = get_risk_free_rate()
        return self.risk_free_rate_value
    
    def get_historical_volatility(self, 
                                symbol: str, 
                                period: str = '1y') -> float:
        """Calculate historical volatility using the stock simulator."""
        cache_key = f"{symbol}_{period}"
        if cache_key in self.volatility_cache:
            return self.volatility_cache[cache_key]
            
        model = GBMModel(ticker=symbol)
        model.calibrate()
        self.volatility_cache[cache_key] = model.volatility
        return model.volatility
    
    def get_bid_ask_spread(self,
                          row: pd.Series) -> Dict[str, float]:
        """Get bid-ask spread for an option.
        
        Args:
            row: A row from the option chain dataframe
            
        Returns:
            Dictionary containing bid, ask, and spread information
        """
        bid = row.get('bid', 0)
        ask = row.get('ask', 0)
        last_price = row.get('lastPrice', 0)
        
        # If bid or ask are zero, estimate them from last price
        if bid == 0 and last_price > 0:
            bid = last_price * 0.95  # 5% below last price
        
        if ask == 0 and last_price > 0:
            ask = last_price * 1.05  # 5% above last price
        
        # Make sure bid and ask are never zero
        if bid == 0 and ask > 0:
            bid = ask * 0.9  # 10% below ask
        elif ask == 0 and bid > 0:
            ask = bid * 1.1  # 10% above bid
        elif bid == 0 and ask == 0 and last_price > 0:
            bid = last_price * 0.95
            ask = last_price * 1.05
        
        # Calculate spread
        spread = ask - bid
        spread_percent = (spread / ((bid + ask) / 2)) * 100 if (bid + ask) > 0 else 0
        
        return {
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'spread_percent': spread_percent
        }
    
    def create_option_contract(self, 
                             row: pd.Series, 
                             symbol: str, 
                             expiration_date: datetime) -> OptionContract:
        """Create an OptionContract from a row of option chain data."""
        # Get bid-ask spread information
        spread_info = self.get_bid_ask_spread(row)
        
        return OptionContract(
            symbol=symbol,
            strike_price=row['strike'],
            expiration_date=expiration_date,
            option_type=row['option_type'],
            premium=row['lastPrice'],
            underlying_price=row['underlyingPrice'],
            implied_volatility=row['impliedVolatility'],
            time_to_expiry=(expiration_date - datetime.now()).days / 365.0,
            risk_free_rate=self.get_risk_free_rate(),
            bid=spread_info['bid'],
            ask=spread_info['ask'],
            spread=spread_info['spread'],
            spread_percent=spread_info['spread_percent']
        )
    
    def get_option_contracts(self, 
                           symbol: str, 
                           expiration_date: Optional[datetime] = None) -> List[OptionContract]:
        """Get a list of OptionContract objects for a symbol."""
        chain = self.get_option_chain(symbol, expiration_date)
        
        if chain.empty:
            return []
        
        if expiration_date is None:
            expiration_date = datetime.strptime(chain['expiration'].iloc[0], '%Y-%m-%d')
        
        contracts = []
        for _, row in chain.iterrows():
            contract = self.create_option_contract(row, symbol, expiration_date)
            contracts.append(contract)
        
        return contracts
    
    def get_atm_options(self, 
                       symbol: str, 
                       expiration_date: Optional[datetime] = None) -> Dict[str, OptionContract]:
        """Get at-the-money call and put options."""
        current_price = self.get_stock_price(symbol)
        chain = self.get_option_chain(symbol, expiration_date)
        
        if chain.empty:
            raise ValueError(f"No options available for {symbol}")
        
        # Use utility function to find ATM options by delta
        atm_call = find_option_by_delta(chain[chain['option_type'] == 'call'], 0.5, current_price, 'call')
        atm_put = find_option_by_delta(chain[chain['option_type'] == 'put'], -0.5, current_price, 'put')
        
        if expiration_date is None:
            expiration_date = datetime.strptime(chain['expiration'].iloc[0], '%Y-%m-%d')
        
        return {
            'call': self.create_option_contract(atm_call, symbol, expiration_date),
            'put': self.create_option_contract(atm_put, symbol, expiration_date)
        }
    
    def find_option_by_delta(self,
                           symbol: str,
                           expiration_date: datetime,
                           option_type: str,
                           target_delta: float) -> OptionContract:
        """
        Find an option with a delta closest to the target delta value.
        
        Args:
            symbol: Ticker symbol
            expiration_date: Option expiration date
            option_type: 'call' or 'put'
            target_delta: Target delta value (positive for calls, negative for puts)
            
        Returns:
            OptionContract with the closest delta to the target
        """
        chain = self.get_option_chain(symbol, expiration_date)
        current_price = self.get_stock_price(symbol)
        
        if chain.empty:
            raise ValueError(f"No options available for {symbol}")
        
        # Filter by option type
        filtered_chain = chain[chain['option_type'] == option_type.lower()]
        
        # Use utility function to find option by delta
        option_row = find_option_by_delta(filtered_chain, target_delta, current_price, option_type.lower())
        
        if option_row is None:
            raise ValueError(f"No suitable {option_type} option found with delta near {target_delta}")
        
        return self.create_option_contract(option_row, symbol, expiration_date) 