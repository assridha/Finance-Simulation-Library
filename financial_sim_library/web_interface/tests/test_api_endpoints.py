"""
Tests for the web interface API endpoints.
"""
import json
import pytest
from flask import url_for

def test_market_data_endpoint(client):
    """Test the market data endpoint."""
    response = client.get('/api/market-data?symbol=AAPL')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'symbol' in data
    assert 'price' in data
    assert 'volatility' in data
    assert data['symbol'] == 'AAPL'

def test_market_data_invalid_symbol(client):
    """Test market data endpoint with invalid symbol."""
    response = client.get('/api/market-data?symbol=INVALID')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'error' in data

def test_option_chain_endpoint(client):
    """Test the option chain endpoint."""
    response = client.get('/api/option-chain?symbol=AAPL')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'calls' in data
    assert 'puts' in data
    assert 'expiry_dates' in data
    assert len(data['calls']) > 0
    assert len(data['puts']) > 0

def test_option_chain_with_expiry(client):
    """Test option chain endpoint with specific expiry date."""
    # First get available expiry dates
    response = client.get('/api/option-chain?symbol=AAPL')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'expiry_dates' in data
    assert len(data['expiry_dates']) > 0
    
    # Use the first available expiry date
    expiry = data['expiry_dates'][0]
    response = client.get(f'/api/option-chain?symbol=AAPL&expiry={expiry}')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert all(opt['expiry'] == expiry for opt in data['calls'])
    assert all(opt['expiry'] == expiry for opt in data['puts'])

def test_simulate_strategy_endpoint(client):
    """Test the strategy simulation endpoint."""
    strategy_config = {
        'symbol': 'AAPL',
        'strategy_type': 'butterfly',
        'legs': [
            {'type': 'call', 'strike': 180, 'position': 1},
            {'type': 'call', 'strike': 190, 'position': -2},
            {'type': 'call', 'strike': 200, 'position': 1}
        ],
        'expiry': '2024-06-21',
        'simulation_params': {
            'paths': 100,
            'time_steps': 50,
            'seed': 42
        }
    }
    
    response = client.post('/api/simulate',
                          data=json.dumps(strategy_config),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'simulation_id' in data
    assert 'status' in data
    assert data['status'] == 'queued'

def test_simulate_invalid_strategy(client):
    """Test simulation endpoint with invalid strategy configuration."""
    invalid_config = {
        'symbol': 'AAPL',
        'strategy_type': 'invalid_strategy'
    }
    
    response = client.post('/api/simulate',
                          data=json.dumps(invalid_config),
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_simulation_results_endpoint(client):
    """Test retrieving simulation results."""
    # First create a simulation
    strategy_config = {
        'symbol': 'AAPL',
        'strategy_type': 'butterfly',
        'legs': [
            {'type': 'call', 'strike': 180, 'position': 1},
            {'type': 'call', 'strike': 190, 'position': -2},
            {'type': 'call', 'strike': 200, 'position': 1}
        ],
        'simulation_params': {
            'paths': 50,
            'time_steps': 25
        }
    }
    
    sim_response = client.post('/api/simulate',
                             data=json.dumps(strategy_config),
                             content_type='application/json')
    sim_data = json.loads(sim_response.data)
    simulation_id = sim_data['simulation_id']
    
    # Then get the results
    response = client.get(f'/api/results/{simulation_id}')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert 'results' in data or 'progress' in data

def test_available_strategies_endpoint(client):
    """Test retrieving available strategy types."""
    response = client.get('/api/strategies')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert 'butterfly' in data
    assert 'iron_condor' in data
    assert all(isinstance(strategy, str) for strategy in data)

def test_user_config_endpoints(client):
    """Test saving and loading user configurations."""
    # Save configuration
    config = {
        'name': 'My Butterfly Strategy',
        'strategy_type': 'butterfly',
        'parameters': {
            'width': 10,
            'center_strike': 190
        }
    }
    
    save_response = client.post('/api/user-config',
                              data=json.dumps(config),
                              content_type='application/json')
    assert save_response.status_code == 201
    save_data = json.loads(save_response.data)
    assert 'config_id' in save_data
    
    # Load configuration
    config_id = save_data['config_id']
    load_response = client.get(f'/api/user-config/{config_id}')
    assert load_response.status_code == 200
    load_data = json.loads(load_response.data)
    assert load_data['name'] == config['name']
    assert load_data['strategy_type'] == config['strategy_type']
    assert load_data['parameters'] == config['parameters']

def test_market_data_rate_limiting(client, app):
    """Test rate limiting on market data endpoint."""
    app.config['TEST_RATE_LIMIT'] = True  # Enable rate limiting for this test
    
    # Make multiple requests in quick succession
    for _ in range(5):
        response = client.get('/api/market-data?symbol=AAPL')
        assert response.status_code == 200
    
    # The next request should be rate limited
    response = client.get('/api/market-data?symbol=AAPL')
    assert response.status_code == 429  # Too Many Requests
    data = json.loads(response.data)
    assert 'error' in data
    assert 'rate limit exceeded' in data['error'].lower() 