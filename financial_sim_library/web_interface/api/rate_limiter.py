"""
Rate limiter implementation.
"""
from time import time
from functools import wraps
from flask import request, jsonify, current_app

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = {}  # {key: (count, start_time)}
        self.window = 60  # 60 seconds window
    
    def _get_key(self):
        """Get the key for rate limiting."""
        if current_app.config.get('TEST_RATE_LIMIT'):
            return 'test_user'
        return request.remote_addr
    
    def _get_limit(self):
        """Get the request limit."""
        if current_app.config.get('TEST_RATE_LIMIT'):
            return 5  # Stricter limit for tests (5 requests allowed, 6th should fail)
        return 100  # Normal limit
    
    def _is_rate_limited(self, key):
        """Check if the request should be rate limited."""
        now = time()
        if key not in self.requests:
            self.requests[key] = (1, now)
            return False
        
        count, start = self.requests[key]
        if now - start >= self.window:
            # Reset if window has passed
            self.requests[key] = (1, now)
            return False
        
        # Check if limit is exceeded before incrementing
        if count >= self._get_limit():
            return True
        
        # Increment count
        self.requests[key] = (count + 1, start)
        return False
    
    def reset(self):
        """Reset all rate limit counters."""
        self.requests.clear()
    
    def limit(self):
        """Decorator to apply rate limiting."""
        def decorator(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                # Skip rate limiting if not in test mode and testing is enabled
                if current_app.config.get('TESTING') and not current_app.config.get('TEST_RATE_LIMIT'):
                    return f(*args, **kwargs)
                
                key = self._get_key()
                if self._is_rate_limited(key):
                    return jsonify({'error': 'Rate limit exceeded'}), 429
                return f(*args, **kwargs)
            return wrapped
        return decorator

# Global rate limiter instance
limiter = RateLimiter() 