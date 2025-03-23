import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from financial_sim_library.option_simulator.data_fetcher import MarketDataFetcher
from financial_sim_library.option_simulator.base import OptionContract

@pytest.fixture
def fetcher():
    return MarketDataFetcher()

@pytest.fixture
def mock_option_chain():
    """Create a mock option chain for testing."""
    # Create sample data
    calls_data = {
        'strike': [100, 105, 110, 115, 120],
        'lastPrice': [10, 7, 4, 2, 1],
        'impliedVolatility': [0.3, 0.35, 0.4, 0.45, 0.5],
        'delta': [0.8, 0.6, 0.4, 0.2, 0.1],
        'gamma': [0.1, 0.15, 0.2, 0.15, 0.1],
        'theta': [-0.1, -0.15, -0.2, -0.15, -0.1],
        'vega': [0.2, 0.25, 0.3, 0.25, 0.2],
        'volume': [100, 200, 300, 200, 100],
        'openInterest': [1000, 2000, 3000, 2000, 1000],
        'bid': [9.9, 6.9, 3.9, 1.9, 0.9],
        'ask': [10.1, 7.1, 4.1, 2.1, 1.1],
        'underlyingPrice': [110] * 5,
        'expiration': ['2024-04-19'] * 5,
        'option_type': ['call'] * 5
    }
    
    puts_data = {
        'strike': [100, 105, 110, 115, 120],
        'lastPrice': [1, 2, 4, 7, 10],
        'impliedVolatility': [0.5, 0.45, 0.4, 0.35, 0.3],
        'delta': [-0.1, -0.2, -0.4, -0.6, -0.8],
        'gamma': [0.1, 0.15, 0.2, 0.15, 0.1],
        'theta': [-0.1, -0.15, -0.2, -0.15, -0.1],
        'vega': [0.2, 0.25, 0.3, 0.25, 0.2],
        'volume': [100, 200, 300, 200, 100],
        'openInterest': [1000, 2000, 3000, 2000, 1000],
        'bid': [0.9, 1.9, 3.9, 6.9, 9.9],
        'ask': [1.1, 2.1, 4.1, 7.1, 10.1],
        'underlyingPrice': [110] * 5,
        'expiration': ['2024-04-19'] * 5,
        'option_type': ['put'] * 5
    }
    
    calls_df = pd.DataFrame(calls_data)
    puts_df = pd.DataFrame(puts_data)
    
    return pd.concat([calls_df, puts_df]).sort_values('strike')

def test_get_stock_price(fetcher):
    """Test getting current stock price."""
    price = fetcher.get_stock_price('AAPL')
    assert price is not None
    assert price > 0

def test_get_option_chain(fetcher):
    """Test getting option chain."""
    chain = fetcher.get_option_chain('AAPL')
    assert not chain.empty
    assert 'strike' in chain.columns
    assert 'option_type' in chain.columns
    assert len(chain[chain['option_type'] == 'call']) > 0
    assert len(chain[chain['option_type'] == 'put']) > 0

def test_get_risk_free_rate(fetcher):
    """Test getting risk-free rate."""
    rate = fetcher.get_risk_free_rate()
    assert rate is not None
    assert rate > 0
    assert rate < 1

def test_get_historical_volatility(fetcher):
    """Test getting historical volatility."""
    vol = fetcher.get_historical_volatility('AAPL')
    assert vol is not None
    assert vol > 0
    assert vol < 1

def test_create_option_contract(fetcher, mock_option_chain):
    """Test creating option contract from chain data."""
    row = mock_option_chain.iloc[0]
    contract = fetcher.create_option_contract(
        row, 
        'AAPL', 
        datetime.strptime(row['expiration'], '%Y-%m-%d')
    )
    assert isinstance(contract, OptionContract)
    assert contract.symbol == 'AAPL'
    assert contract.strike_price == row['strike']
    assert contract.option_type == row['option_type']
    assert contract.premium == row['lastPrice']

def test_get_option_contracts(fetcher):
    """Test getting list of option contracts."""
    contracts = fetcher.get_option_contracts('AAPL')
    assert len(contracts) > 0
    assert all(isinstance(c, OptionContract) for c in contracts)
    assert all(c.symbol == 'AAPL' for c in contracts)

def test_get_atm_options(fetcher):
    """Test getting ATM options."""
    atm_options = fetcher.get_atm_options('AAPL')
    assert 'call' in atm_options
    assert 'put' in atm_options
    assert isinstance(atm_options['call'], OptionContract)
    assert isinstance(atm_options['put'], OptionContract)
    assert atm_options['call'].option_type == 'call'
    assert atm_options['put'].option_type == 'put'

def test_get_option_strategy_contracts_covered_call(fetcher):
    """Test getting covered call strategy contracts."""
    strategy = fetcher.get_option_strategy_contracts(
        'AAPL', 
        'covered_call',
        target_delta=0.3
    )
    assert 'stock' in strategy
    assert 'call' in strategy
    assert isinstance(strategy['call'], OptionContract)
    assert strategy['call'].option_type == 'call'

def test_get_option_strategy_contracts_pmcc(fetcher):
    """Test getting poor man's covered call strategy contracts."""
    strategy = fetcher.get_option_strategy_contracts(
        'AAPL', 
        'poor_mans_covered_call',
        target_delta=0.3
    )
    assert 'long_call' in strategy
    assert 'short_call' in strategy
    assert isinstance(strategy['long_call'], OptionContract)
    assert isinstance(strategy['short_call'], OptionContract)
    assert strategy['long_call'].option_type == 'call'
    assert strategy['short_call'].option_type == 'call'

def test_get_option_strategy_contracts_vertical_spread(fetcher):
    """Test getting vertical spread strategy contracts."""
    strategy = fetcher.get_option_strategy_contracts(
        'AAPL', 
        'vertical_spread',
        target_delta=0.3
    )
    assert 'long_call' in strategy
    assert 'short_call' in strategy
    assert isinstance(strategy['long_call'], OptionContract)
    assert isinstance(strategy['short_call'], OptionContract)
    assert strategy['long_call'].option_type == 'call'
    assert strategy['short_call'].option_type == 'call'

def test_get_option_strategy_contracts_invalid_strategy(fetcher):
    """Test handling of invalid strategy type."""
    with pytest.raises(ValueError):
        fetcher.get_option_strategy_contracts('AAPL', 'invalid_strategy')

def test_get_option_chain_with_expiration(fetcher):
    """Test getting option chain with specific expiration date."""
    # Get available expiration dates
    chain = fetcher.get_option_chain('AAPL')
    if not chain.empty:
        exp_date = datetime.strptime(chain['expiration'].iloc[0], '%Y-%m-%d')
        chain_with_exp = fetcher.get_option_chain('AAPL', expiration_date=exp_date)
        assert not chain_with_exp.empty
        assert all(chain_with_exp['expiration'] == exp_date.strftime('%Y-%m-%d')) 