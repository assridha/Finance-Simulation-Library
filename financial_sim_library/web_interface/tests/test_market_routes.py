"""
Tests for market data and simulation endpoints.
"""
import json
import pytest
from unittest.mock import patch

def test_get_historical_data(client):
    """Test getting historical data for a valid symbol."""
    response = client.get('/api/market/historical-data?symbol=AAPL')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'timestamps' in data
    assert 'open' in data
    assert 'high' in data
    assert 'low' in data
    assert 'close' in data
    assert 'volume' in data
    assert len(data['timestamps']) > 0
    assert len(data['open']) == len(data['timestamps'])

def test_get_historical_data_invalid_symbol(client):
    """Test getting historical data for an invalid symbol."""
    response = client.get('/api/market/historical-data?symbol=INVALID')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'error' in data

def test_get_historical_data_missing_symbol(client):
    """Test getting historical data without providing a symbol."""
    response = client.get('/api/market/historical-data')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Symbol is required' in data['error']

def test_get_historical_data_with_params(client):
    """Test getting historical data with custom period and interval."""
    response = client.get('/api/market/historical-data?symbol=AAPL&period=1y&interval=1wk')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data['timestamps']) > 0

def test_run_simulation(client):
    """Test running a price simulation for a valid symbol."""
    config = {
        'symbol': 'AAPL',
        'days_to_simulate': 30,
        'num_simulations': 100,
        'growth_rate': 0.1
    }
    
    response = client.post('/api/market/simulate',
                          data=json.dumps(config),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'current_price' in data
    assert 'simulated_paths' in data
    assert 'statistics' in data
    assert 'timestamps' in data
    assert len(data['simulated_paths']) == 100
    assert len(data['timestamps']) == 31  # days_to_simulate + 1

def test_run_simulation_invalid_symbol(client):
    """Test running a simulation for an invalid symbol."""
    config = {
        'symbol': 'INVALID',
        'days_to_simulate': 30,
        'num_simulations': 100
    }
    
    response = client.post('/api/market/simulate',
                          data=json.dumps(config),
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_run_simulation_missing_symbol(client):
    """Test running a simulation without providing a symbol."""
    config = {
        'days_to_simulate': 30,
        'num_simulations': 100
    }
    
    response = client.post('/api/market/simulate',
                          data=json.dumps(config),
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Symbol is required' in data['error']

def test_run_simulation_invalid_content_type(client):
    """Test running a simulation with invalid content type."""
    response = client.post('/api/market/simulate',
                          data='not json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Content type must be application/json' in data['error']

def test_run_simulation_rate_limit(client, app):
    """Test rate limiting on simulation endpoint."""
    app.config['TEST_RATE_LIMIT'] = True  # Enable rate limiting for this test
    
    config = {
        'symbol': 'AAPL',
        'days_to_simulate': 30,
        'num_simulations': 100
    }
    
    # Make multiple requests to trigger rate limit
    for _ in range(5):
        response = client.post('/api/market/simulate',
                             data=json.dumps(config),
                             content_type='application/json')
        assert response.status_code == 200
    
    # The next request should be rate limited
    response = client.post('/api/market/simulate',
                          data=json.dumps(config),
                          content_type='application/json')
    assert response.status_code == 429  # Rate limit reached
    data = json.loads(response.data)
    assert 'error' in data
    assert 'rate limit exceeded' in data['error'].lower() 