# Create a new file: utils/logging_config.py

import logging
import os
from datetime import datetime

def setup_logger(name: str, level: str = None) -> logging.Logger:
    """Setup logger with consistent formatting"""
    
    # Determine log level
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if os.getenv('LOG_TO_FILE', 'false').lower() == 'true':
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(
            f'{log_dir}/{name}_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Usage example for your modules:
def get_scraper_logger():
    return setup_logger('letterboxd.scraper')

def get_model_logger():
    return setup_logger('letterboxd.model')

def get_api_logger():
    return setup_logger('letterboxd.api')