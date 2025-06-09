import os
import pymongo
from typing import Tuple, Optional
from config import Config
from utils.logging_config import setup_logger

logger = setup_logger('letterboxd.database')

class DatabaseConnection:
    """Enhanced database connection manager"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.config = None
    
    def connect(self) -> Tuple[str, pymongo.MongoClient, str]:
        """
        Connect to MongoDB with improved error handling
        Returns: (db_name, client, tmdb_key)
        """
        try:
            self.config = Config.get_db_config()
            
            # Validate required config
            if not self.config['db_name']:
                raise ValueError("Database name not configured")
            
            if not self.config['tmdb_key']:
                raise ValueError("TMDB API key not configured")
            
            # Create connection
            if self.config['connection_url']:
                # Direct connection URL
                self.client = pymongo.MongoClient(
                    self.config['connection_url'], 
                    server_api=pymongo.server_api.ServerApi('1'),
                    connectTimeoutMS=10000,  # 10 second timeout
                    serverSelectionTimeoutMS=10000
                )
            else:
                # Atlas connection
                if not all([self.config['username'], self.config['password'], self.config['cluster_id']]):
                    raise ValueError("MongoDB Atlas credentials incomplete")
                
                connection_string = (
                    f"mongodb+srv://{self.config['username']}:{self.config['password']}"
                    f"@cluster0.{self.config['cluster_id']}.mongodb.net/"
                    f"{self.config['db_name']}?retryWrites=true&w=majority"
                )
                
                self.client = pymongo.MongoClient(
                    connection_string,
                    server_api=pymongo.server_api.ServerApi('1'),
                    connectTimeoutMS=10000,
                    serverSelectionTimeoutMS=10000
                )
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.config['db_name']]
            
            logger.info(f"Successfully connected to database: {self.config['db_name']}")
            
            return self.config['db_name'], self.client, self.config['tmdb_key']
            
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")
    
    def get_collection(self, collection_name: str):
        """Get a collection with error handling"""
        if not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        return self.db[collection_name]
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self.db, self.tmdb_key
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        if exc_type:
            logger.error(f"Database operation failed: {exc_val}")
        return False

# Backwards compatibility function
def connect_to_db() -> Tuple[str, pymongo.MongoClient, str]:
    """
    Legacy function for backwards compatibility
    """
    db_conn = DatabaseConnection()
    return db_conn.connect()

# Context manager for database operations
class DatabaseContext:
    """Context manager for database operations"""
    
    def __init__(self):
        self.db_conn = DatabaseConnection()
        self.db_name = None
        self.client = None
        self.tmdb_key = None
    
    def __enter__(self):
        self.db_name, self.client, self.tmdb_key = self.db_conn.connect()
        return self.db_conn.db, self.tmdb_key
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db_conn.close()
        if exc_type:
            logger.error(f"Database operation failed: {exc_val}")
        return False