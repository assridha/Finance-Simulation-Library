"""
Option chain routes.
"""
from flask import Blueprint, jsonify, request, current_app
from ..services.option_service import OptionService
from ..app import cache
from .rate_limiter import limiter

option_bp = Blueprint('option', __name__)
option_service = OptionService()

@option_bp.route('/option-chain', methods=['GET'])
@limiter.limit()
@cache.cached(timeout=300, query_string=True)  # Cache for 5 minutes
def get_option_chain():
    """Get option chain data for a symbol."""
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
    
    expiry = request.args.get('expiry')  # Optional parameter
    
    try:
        data = option_service.get_option_chain(symbol, expiry)
        return jsonify(data)
    except ValueError as e:
        error_msg = str(e)
        if "invalid symbol" in error_msg.lower():
            return jsonify({'error': error_msg}), 400
        else:
            return jsonify({'error': error_msg}), 404
    except Exception as e:
        current_app.logger.error(f"Error fetching option chain: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500 