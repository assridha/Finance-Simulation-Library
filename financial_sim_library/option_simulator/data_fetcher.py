import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from .base import OptionContract
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
        self.cache: Dict[str, pd.DataFrame] = {}
    
    def get_stock_price(self, symbol: str) -> float:
        """Get current stock price."""
        return fetch_current_price(symbol)
    
    def get_option_chain(self, 
                        symbol: str, 
                        expiration_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get option chain for a symbol."""
        # Convert datetime to string format if provided
        exp_date_str = expiration_date.strftime('%Y-%m-%d') if expiration_date else None
        
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
        
        return chain
    
    def get_risk_free_rate(self) -> float:
        """Get current risk-free rate."""
        return get_risk_free_rate()
    
    def get_historical_volatility(self, 
                                symbol: str, 
                                period: str = '1y') -> float:
        """Calculate historical volatility using the stock simulator."""
        model = GBMModel(ticker=symbol)
        model.calibrate()
        return model.volatility
    
    def create_option_contract(self, 
                             row: pd.Series, 
                             symbol: str, 
                             expiration_date: datetime) -> OptionContract:
        """Create an OptionContract from a row of option chain data."""
        return OptionContract(
            symbol=symbol,
            strike_price=row['strike'],
            expiration_date=expiration_date,
            option_type=row['option_type'],
            premium=row['lastPrice'],
            underlying_price=row['underlyingPrice'],
            implied_volatility=row['impliedVolatility'],
            time_to_expiry=(expiration_date - datetime.now()).days / 365.0,
            risk_free_rate=self.get_risk_free_rate()
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
    
    def get_option_strategy_contracts(self, 
                                    symbol: str, 
                                    strategy_type: str, 
                                    expiration_date: Optional[datetime] = None,
                                    target_delta: float = 0.3) -> Dict:
        """Get option contracts for a specific strategy."""
        current_price = self.get_stock_price(symbol)
        chain = self.get_option_chain(symbol, expiration_date)
        
        if chain.empty:
            raise ValueError(f"No options available for {symbol}")
        
        if expiration_date is None:
            expiration_date = datetime.strptime(chain['expiration'].iloc[0], '%Y-%m-%d')
        
        if strategy_type == 'covered_call':
            # Get OTM call with target delta for covered call
            otm_call = find_option_by_delta(
                chain[chain['option_type'] == 'call'], 
                target_delta, 
                current_price, 
                'call'
            )
            return {
                'stock': current_price,
                'call': self.create_option_contract(otm_call, symbol, expiration_date)
            }
        
        elif strategy_type == 'poor_mans_covered_call':
            # Get deep ITM call and OTM call
            itm_call = find_option_by_delta(
                chain[chain['option_type'] == 'call'], 
                0.8,  # Deep ITM call
                current_price, 
                'call'
            )
            otm_call = find_option_by_delta(
                chain[chain['option_type'] == 'call'], 
                target_delta, 
                current_price, 
                'call'
            )
            
            return {
                'long_call': self.create_option_contract(itm_call, symbol, expiration_date),
                'short_call': self.create_option_contract(otm_call, symbol, expiration_date)
            }
        
        elif strategy_type == 'vertical_spread':
            # Get ATM call and OTM call for vertical spread
            atm_call = find_option_by_delta(
                chain[chain['option_type'] == 'call'], 
                0.5,  # ATM call
                current_price, 
                'call'
            )
            otm_call = find_option_by_delta(
                chain[chain['option_type'] == 'call'], 
                target_delta, 
                current_price, 
                'call'
            )
            
            return {
                'long_call': self.create_option_contract(atm_call, symbol, expiration_date),
                'short_call': self.create_option_contract(otm_call, symbol, expiration_date)
            }
        
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}") 