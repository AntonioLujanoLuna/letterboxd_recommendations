import os
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ScrapingConfig:
    """Scraping-specific configuration"""
    max_pages_per_user: int = 128
    chunk_size_movies: int = 12
    chunk_size_ratings: int = 10
    request_delay: float = 0.5  # seconds between requests
    max_retries: int = 5
    concurrent_requests: int = 10  # Control concurrency
    timeout_seconds: int = 30
    
    @classmethod
    def from_env(cls):
        return cls(
            max_pages_per_user=int(os.getenv('MAX_PAGES_PER_USER', 128)),
            chunk_size_movies=int(os.getenv('CHUNK_SIZE_MOVIES', 12)),
            chunk_size_ratings=int(os.getenv('CHUNK_SIZE_RATINGS', 10)),
            request_delay=float(os.getenv('REQUEST_DELAY', 0.5)),
            max_retries=int(os.getenv('MAX_RETRIES', 5)),
            concurrent_requests=int(os.getenv('CONCURRENT_REQUESTS', 10)),
            timeout_seconds=int(os.getenv('TIMEOUT_SECONDS', 30))
        )

@dataclass
class ModelConfig:
    """Model-specific configuration"""
    popularity_thresholds_500k: List[int] = None
    default_training_size: int = 500000
    max_training_size: int = 1200000
    min_review_threshold: int = 20
    svd_factors: int = 100
    svd_epochs: int = 20
    svd_learning_rate: float = 0.005
    svd_regularization: float = 0.02
    
    # User sampling settings
    samples_per_bucket: Dict[str, int] = None
    
    def __post_init__(self):
        if self.popularity_thresholds_500k is None:
            self.popularity_thresholds_500k = [2500, 2000, 1500, 1000, 700, 400, 250, 150]
        
        if self.samples_per_bucket is None:
            self.samples_per_bucket = {
                'casual': 1000,      # 10-50 reviews
                'regular': 2000,     # 50-200 reviews  
                'active': 3000,      # 200-500 reviews
                'power': 3000,       # 500-1000 reviews
                'super': 1000        # 1000+ reviews
            }

@dataclass
class RedisConfig:
    """Redis-specific configuration"""
    url: str = 'redis://localhost:6379'
    queues: List[str] = None
    job_result_ttl: int = 45  # seconds
    job_ttl: int = 200  # seconds
    
    def __post_init__(self):
        if self.queues is None:
            self.queues = ['high', 'default', 'low']

@dataclass  
class ValidationConfig:
    """Validation settings"""
    min_year: int = 1800
    max_year_offset: int = 5  # years in the future allowed
    max_username_length: int = 50
    max_movie_title_length: int = 500
    max_genres_per_movie: int = 10

class Config:
    """Enhanced centralized configuration management"""
    
    def __init__(self):
        self.scraping = ScrapingConfig.from_env()
        self.model = ModelConfig()
        self.redis = RedisConfig()
        self.validation = ValidationConfig()
        
        # Override Redis URL from environment
        self.redis.url = os.getenv('REDISCLOUD_URL', self.redis.url)
        self.redis.job_result_ttl = int(os.getenv('JOB_RESULT_TTL', self.redis.job_result_ttl))
        self.redis.job_ttl = int(os.getenv('JOB_TTL', self.redis.job_ttl))
    
    def get_db_config(self) -> Dict[str, Any]:
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
    
    @classmethod
    def validate_username(cls, username: str) -> bool:
        """Validate username for safety"""
        if not username or not isinstance(username, str):
            return False
        
        # Remove potentially dangerous characters
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
        return (all(c in allowed_chars for c in username) and 
                len(username) <= cls().validation.max_username_length and
                len(username.strip()) > 0)
    
    @classmethod
    def validate_year(cls, year: Optional[int]) -> bool:
        """Validate movie year"""
        if year is None:
            return True
        
        config = cls()
        max_year = datetime.now().year + config.validation.max_year_offset
        return config.validation.min_year <= year <= max_year
    
    @classmethod
    def validate_movie_title(cls, title: str) -> bool:
        """Validate movie title"""
        if not title or not isinstance(title, str):
            return False
        
        return len(title.strip()) <= cls().validation.max_movie_title_length
    
    @classmethod 
    def validate_rating(cls, rating: float) -> bool:
        """Validate rating value"""
        return isinstance(rating, (int, float)) and -1 <= rating <= 10
    
    def log_configuration(self):
        """Log current configuration for debugging"""
        print("=== Configuration Summary ===")
        print(f"Scraping delay: {self.scraping.request_delay}s")
        print(f"Max retries: {self.scraping.max_retries}")
        print(f"Concurrent requests: {self.scraping.concurrent_requests}")
        print(f"Model training size: {self.model.default_training_size}")
        print(f"Redis queues: {self.redis.queues}")
        print(f"Job TTL: {self.redis.job_ttl}s")
        print("============================")

# Global config instance
config = Config()

# Backwards compatibility constants
MAX_PAGES_PER_USER = config.scraping.max_pages_per_user
CHUNK_SIZE_MOVIES = config.scraping.chunk_size_movies
CHUNK_SIZE_RATINGS = config.scraping.chunk_size_ratings
REQUEST_DELAY = config.scraping.request_delay
MAX_RETRIES = config.scraping.max_retries
POPULARITY_THRESHOLDS_500K = config.model.popularity_thresholds_500k
DEFAULT_TRAINING_SIZE = config.model.default_training_size
MAX_TRAINING_SIZE = config.model.max_training_size
MIN_REVIEW_THRESHOLD = config.model.min_review_threshold