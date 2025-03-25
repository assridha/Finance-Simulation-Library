"""
Test configuration.
"""
import pytest
from financial_sim_library.web_interface.app import create_app
from financial_sim_library.web_interface.api.rate_limiter import limiter

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    app = create_app({
        'TESTING': True,
        'TEST_RATE_LIMIT': False,  # Disable rate limiting by default
        'RATELIMIT_ENABLED': True,
        'RATELIMIT_STORAGE_URL': 'memory://',
        'RATELIMIT_STORAGE_BACKEND': 'memory',
        'RATELIMIT_RESET_TIME': 60,
        'SESSION_TYPE': 'filesystem',
        'CACHE_TYPE': 'SimpleCache',
        'SECRET_KEY': 'test'
    })
    
    yield app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Create a test CLI runner for the Flask application."""
    return app.test_cli_runner()

@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    limiter.reset()
    yield 