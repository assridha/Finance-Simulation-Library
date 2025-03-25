"""
Flask application factory.
"""
from flask import Flask
from flask_cors import CORS
from .extensions import cache, session
from .api.market_routes import market_bp
from .api.option_routes import option_bp
from .api.routes import bp as api_bp

def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Default configuration
    app.config.setdefault('SECRET_KEY', 'dev')
    app.config.setdefault('SESSION_TYPE', 'filesystem')
    app.config.setdefault('CACHE_TYPE', 'SimpleCache')
    app.config.setdefault('CACHE_DEFAULT_TIMEOUT', 300)
    
    # Apply configuration
    if config:
        app.config.update(config)
    
    # Initialize extensions
    cache.init_app(app)
    session.init_app(app)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(market_bp, url_prefix='/api')
    app.register_blueprint(option_bp, url_prefix='/api')
    
    return app 