import os
from typing import List, Optional

class Config:
    """Centralized configuration management"""
    
    # Scraping settings
    MAX_PAGES_PER_USER = 128
    CHUNK_SIZE_MOVIES = 12
    CHUNK_SIZE_RATINGS = 10
    REQUEST_DELAY = 0.5  # seconds between requests
    MAX_RETRIES = 5
    
    # Model settings
    POPULARITY_THRESHOLDS_500K = [2500, 2000, 1500, 1000, 700, 400, 250, 150]
    DEFAULT_TRAINING_SIZE = 500000
    MAX_TRAINING_SIZE = 1200000
    MIN_REVIEW_THRESHOLD = 20
    
    # User sampling settings
    SAMPLES_PER_BUCKET = {
        'casual': 1000,      # 10-50 reviews
        'regular': 2000,     # 50-200 reviews  
        'active': 3000,      # 200-500 reviews
        'power': 3000,       # 500-1000 reviews
        'super': 1000        # 1000+ reviews
    }
    
    # Database settings
    @staticmethod
    def get_db_config():
        """Get database configuration from environment or config file"""
        try:
            # Try to import from local config file first
            if os.getcwd().endswith("data_processing"):
                from db_config import config, tmdb_key
            else:
                from data_processing.db_config import config, tmdb_key
            
            return {
                'db_name': config["MONGO_DB"],
                'connection_url': config.get("CONNECTION_URL"),
                'username': config.get("MONGO_USERNAME"),
                'password': config.get("MONGO_PASSWORD"), 
                'cluster_id': config.get("MONGO_CLUSTER_ID"),
                'tmdb_key': tmdb_key
            }
        except (ImportError, ModuleNotFoundError):
            # Fallback to environment variables
            return {
                'db_name': os.environ.get('MONGO_DB'),
                'connection_url': os.environ.get("CONNECTION_URL"),
                'username': os.environ.get("MONGO_USERNAME"),
                'password': os.environ.get("MONGO_PASSWORD"),
                'cluster_id': os.environ.get("MONGO_CLUSTER_ID"),
                'tmdb_key': os.environ.get('TMDB_KEY')
            }
    
    # Redis settings
    REDIS_URL = os.getenv('REDISCLOUD_URL', 'redis://localhost:6379')
    REDIS_QUEUES = ['high', 'default', 'low']
    
    # Job settings
    JOB_RESULT_TTL = 45  # seconds
    JOB_TTL = 200  # seconds
    
    # Validation settings
    MIN_YEAR = 1800
    MAX_YEAR_OFFSET = 5  # years in the future allowed
    
    @classmethod
    def validate_username(cls, username: str) -> bool:
        """Validate username for safety"""
        if not username or not isinstance(username, str):
            return False
        
        # Remove potentially dangerous characters
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
        return all(c in allowed_chars for c in username) and len(username) <= 50
    
    @classmethod
    def validate_year(cls, year: Optional[int]) -> bool:
        """Validate movie year"""
        if year is None:
            return True
        
        import datetime
        max_year = datetime.datetime.now().year + cls.MAX_YEAR_OFFSET
        return cls.MIN_YEAR <= year <= max_year