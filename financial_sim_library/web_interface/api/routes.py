"""
API routes for the web interface.
"""
import uuid
from flask import jsonify, request, current_app
from . import bp
from ..extensions import cache
from .rate_limiter import limiter
from ..services.simulation_service import SimulationService
from ..services.plot_service import PlotService

simulation_service = SimulationService()
plot_service = PlotService()

@bp.route('/market-data', methods=['GET'])
@limiter.limit()
@cache.cached(timeout=60, query_string=True)
def get_market_data():
    """Get current market data for a symbol."""
    symbol = request.args.get('symbol', '').upper()
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
        
    try:
        data = simulation_service.get_market_data(symbol)
        return jsonify(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404

@bp.route('/option-chain', methods=['GET'])
@limiter.limit()
@cache.cached(timeout=60, query_string=True)
def get_option_chain():
    """Get option chain data for a symbol."""
    symbol = request.args.get('symbol', '').upper()
    expiry = request.args.get('expiry')
    
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
        
    try:
        data = simulation_service.get_option_chain(symbol, expiry)
        return jsonify(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404

@bp.route('/simulate', methods=['POST'])
@limiter.limit()
def simulate_strategy():
    """Start a new strategy simulation."""
    if not request.is_json:
        return jsonify({'error': 'Content type must be application/json'}), 400
        
    config = request.get_json()
    
    try:
        # Validate configuration
        simulation_service.validate_strategy_config(config)
        
        # Generate unique ID for this simulation
        simulation_id = str(uuid.uuid4())
        
        # Queue the simulation
        simulation_service.queue_simulation(simulation_id, config)
        
        return jsonify({
            'simulation_id': simulation_id,
            'status': 'queued'
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/results/<simulation_id>', methods=['GET'])
def get_simulation_results(simulation_id):
    """Get results of a simulation by ID."""
    try:
        results = simulation_service.get_simulation_results(simulation_id)
        return jsonify(results)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404

@bp.route('/strategies', methods=['GET'])
@cache.cached(timeout=3600)  # Cache for 1 hour
def get_available_strategies():
    """Get list of available strategy types."""
    strategies = simulation_service.get_available_strategies()
    return jsonify(strategies)

@bp.route('/user-config', methods=['POST'])
@limiter.limit()
def save_user_config():
    """Save a user's strategy configuration."""
    if not request.is_json:
        return jsonify({'error': 'Content type must be application/json'}), 400
        
    config = request.get_json()
    
    try:
        config_id = simulation_service.save_user_config(config)
        return jsonify({'config_id': config_id}), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/user-config/<config_id>', methods=['GET'])
def get_user_config(config_id):
    """Get a saved user configuration."""
    try:
        config = simulation_service.get_user_config(config_id)
        return jsonify(config)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404

@bp.route('/plot/<simulation_id>/<plot_type>', methods=['GET'])
def get_plot(simulation_id, plot_type):
    """Get a specific plot for a simulation."""
    try:
        plot_data = plot_service.get_plot(simulation_id, plot_type)
        return jsonify(plot_data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404 