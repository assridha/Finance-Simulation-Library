"""
Configuration settings for the web interface.
"""
import os
from datetime import timedelta

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    CACHE_TYPE = "simple"
    CACHE_DEFAULT_TIMEOUT = 300
    SESSION_TYPE = "filesystem"
    PERMANENT_SESSION_LIFETIME = timedelta(days=31)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # API Rate limiting
    RATELIMIT_DEFAULT = "100/hour"
    RATELIMIT_STORAGE_URL = "memory://"
    
    # Simulation settings
    MAX_SIMULATION_PATHS = 10000
    DEFAULT_SIMULATION_PATHS = 1000
    MAX_TIME_STEPS = 252  # One trading year
    
    # Job Queue settings
    JOBS_QUEUE_MAXSIZE = 100
    JOB_TIMEOUT = 300  # 5 minutes

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False
    
class TestConfig(Config):
    """Test configuration."""
    TESTING = True
    DEBUG = False
    WTF_CSRF_ENABLED = False
    SERVER_NAME = "localhost.localdomain"
    
    # Use in-memory storage for testing
    SESSION_TYPE = "filesystem"
    CACHE_TYPE = "simple"
    
    # Reduce limits for faster testing
    MAX_SIMULATION_PATHS = 100
    DEFAULT_SIMULATION_PATHS = 50
    
class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    # Override these in production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Use Redis for caching in production
    CACHE_TYPE = "redis"
    SESSION_TYPE = "redis"
    
    # Stricter rate limiting
    RATELIMIT_DEFAULT = "60/hour" 