"""
Flask extensions.
"""
from flask_caching import Cache
from flask_session import Session

# Initialize extensions
cache = Cache()
session = Session() 