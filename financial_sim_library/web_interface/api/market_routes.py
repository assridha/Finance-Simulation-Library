"""
Market data routes.
"""
from flask import Blueprint, jsonify, request, current_app
import json
from ..services.market_service import MarketService
from ..app import cache
from .rate_limiter import limiter

market_bp = Blueprint('market', __name__)
market_service = MarketService()

@market_bp.route('/market-data', methods=['GET'])
@limiter.limit()
@cache.cached(timeout=300, query_string=True)  # Cache for 5 minutes
def get_market_data():
    """Get current market data for a symbol."""
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
    
    try:
        data = market_service.get_market_data(symbol)
        return jsonify(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@market_bp.route('/market-data/historical', methods=['GET'])
@limiter.limit()
@cache.cached(timeout=300, query_string=True)  # Cache for 5 minutes
def get_historical_data():
    """Get historical price data for a symbol."""
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
    
    # Get optional parameters
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    
    try:
        data = market_service.get_historical_data(
            symbol=symbol,
            period=period,
            interval=interval
        )
        return jsonify(data)
    except ValueError as e:
        if 'Invalid symbol' in str(e):
            return jsonify({'error': str(e)}), 404
        return jsonify({'error': str(e)}), 400

@market_bp.route('/market/simulate', methods=['POST'])  # Changed from /simulate to /market/simulate
@limiter.limit()
def run_simulation():
    """Run a Monte Carlo price simulation."""
    if not request.is_json:
        return jsonify({'error': 'Content type must be application/json'}), 400
    
    data = request.get_json()
    if not data or 'symbol' not in data:
        return jsonify({'error': 'Symbol is required'}), 400
    
    try:
        days = int(data.get('days_to_simulate', 252))
        sims = int(data.get('num_simulations', 1000))
        results = market_service.run_price_simulation(data['symbol'], days, sims)
        return jsonify(results)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error running simulation: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500 