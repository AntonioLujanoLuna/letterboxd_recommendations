import os
import pymongo
import threading
from typing import Tuple, Optional
from contextlib import contextmanager
from config import Config

class DatabaseManager:
    """Singleton database manager with connection pooling"""
    
    _instance = None
    _lock = threading.Lock()
    _client = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._initialize_connection()
    
    def _get_config(self):
        """Get database configuration from Config class"""
        config_instance = Config()
        return config_instance.get_db_config()
    
    def _initialize_connection(self):
        """Initialize MongoDB connection with proper settings"""
        try:
            self._config = self._get_config()
            
            if not self._config['db_name']:
                raise ValueError("Database name not configured")
            
            if not self._config['tmdb_key']:
                raise ValueError("TMDB API key not configured")
            
            # Connection settings for production
            connection_kwargs = {
                'server_api': pymongo.server_api.ServerApi('1'),
                'connectTimeoutMS': 10000,
                'serverSelectionTimeoutMS': 10000,
                'maxPoolSize': 50,  # Connection pooling
                'minPoolSize': 5,
                'maxIdleTimeMS': 30000,
                'retryWrites': True,
                'w': 'majority'
            }
            
            if self._config['connection_url']:
                # Direct connection URL
                self._client = pymongo.MongoClient(
                    self._config['connection_url'], 
                    **connection_kwargs
                )
            else:
                # Atlas connection
                if not all([self._config['username'], self._config['password'], self._config['cluster_id']]):
                    raise ValueError("MongoDB Atlas credentials incomplete")
                
                connection_string = (
                    f"mongodb+srv://{self._config['username']}:{self._config['password']}"
                    f"@cluster0.{self._config['cluster_id']}.mongodb.net/"
                    f"{self._config['db_name']}?retryWrites=true&w=majority"
                )
                
                self._client = pymongo.MongoClient(connection_string, **connection_kwargs)
            
            # Test connection
            self._client.admin.command('ping')
            print(f"Successfully connected to database: {self._config['db_name']}")
            
        except Exception as e:
            print(f"Database connection failed: {str(e)}")
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")
    
    @property
    def client(self) -> pymongo.MongoClient:
        """Get MongoDB client"""
        if self._client is None:
            self._initialize_connection()
        return self._client
    
    @property
    def db(self):
        """Get database instance"""
        return self.client[self._config['db_name']]
    
    @property
    def tmdb_key(self) -> str:
        """Get TMDB API key"""
        return self._config['tmdb_key']
    
    def get_collection(self, collection_name: str):
        """Get collection with error handling"""
        try:
            return self.db[collection_name]
        except Exception as e:
            print(f"Error accessing collection {collection_name}: {str(e)}")
            raise
    
    def close(self):
        """Close database connection"""
        if self._client:
            self._client.close()
            self._client = None
            print("Database connection closed")

# Global instance
_db_manager = None

def get_db_manager():
    """Get the singleton database manager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

# Context manager for database operations
@contextmanager
def get_database():
    """Context manager for database operations"""
    db_manager = get_db_manager()
    try:
        yield db_manager.db, db_manager.tmdb_key
    except Exception as e:
        print(f"Database operation failed: {str(e)}")
        raise

# Backwards compatibility function
def connect_to_db() -> Tuple[str, pymongo.MongoClient, str]:
    """Legacy function for backwards compatibility"""
    db_manager = get_db_manager()
    return db_manager._config['db_name'], db_manager.client, db_manager.tmdb_key

# Enhanced DatabaseConnection class for legacy compatibility
class DatabaseConnection:
    """Enhanced database connection manager for legacy compatibility"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
    
    def connect(self) -> Tuple[str, pymongo.MongoClient, str]:
        return connect_to_db()
    
    def get_collection(self, collection_name: str):
        return self.db_manager.get_collection(collection_name)
    
    def close(self):
        pass  # Connection pooling handles this
    
    def __enter__(self):
        return self.db_manager.db, self.db_manager.tmdb_key
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"Database operation failed: {exc_val}")
        return False