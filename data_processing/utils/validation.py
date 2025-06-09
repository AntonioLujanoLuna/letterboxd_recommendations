from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
from config import Config

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class DataValidator:
    """Utility class for data validation"""
    
    @staticmethod
    def validate_user_data(user_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate and clean user rating data
        Returns: (cleaned_data, error_messages)
        """
        cleaned_data = []
        errors = []
        
        for i, rating in enumerate(user_data):
            try:
                # Validate required fields
                if 'movie_id' not in rating or 'rating_val' not in rating or 'user_id' not in rating:
                    errors.append(f"Rating {i}: Missing required fields")
                    continue
                
                # Validate movie_id format
                if not isinstance(rating['movie_id'], str) or len(rating['movie_id']) == 0:
                    errors.append(f"Rating {i}: Invalid movie_id")
                    continue
                
                # Validate rating value
                rating_val = rating['rating_val']
                if not isinstance(rating_val, (int, float)) or rating_val < -1 or rating_val > 10:
                    errors.append(f"Rating {i}: Invalid rating value {rating_val}")
                    continue
                
                # Validate user_id
                if not Config.validate_username(rating['user_id']):
                    errors.append(f"Rating {i}: Invalid user_id")
                    continue
                
                cleaned_data.append(rating)
                
            except Exception as e:
                errors.append(f"Rating {i}: Validation error - {str(e)}")
        
        return cleaned_data, errors
    
    @staticmethod
    def validate_movie_data(movie_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate and clean movie data
        Returns: (cleaned_data, error_messages)
        """
        cleaned_data = movie_data.copy()
        errors = []
        
        try:
            # Validate year
            if 'year_released' in cleaned_data:
                year = cleaned_data['year_released']
                if year is not None and not Config.validate_year(year):
                    errors.append(f"Invalid year: {year}")
                    cleaned_data['year_released'] = None
            
            # Clean and validate movie title
            if 'movie_title' in cleaned_data:
                title = cleaned_data['movie_title']
                if title:
                    # Remove excessive whitespace
                    cleaned_data['movie_title'] = re.sub(r'\s+', ' ', str(title).strip())
                    if len(cleaned_data['movie_title']) > 500:  # Reasonable limit
                        cleaned_data['movie_title'] = cleaned_data['movie_title'][:500]
            
            # Validate IMDB ID format
            if 'imdb_id' in cleaned_data and cleaned_data['imdb_id']:
                imdb_id = cleaned_data['imdb_id']
                if not re.match(r'^tt\d+$', imdb_id):
                    errors.append(f"Invalid IMDB ID format: {imdb_id}")
                    cleaned_data['imdb_id'] = ''
            
            # Validate TMDB ID format
            if 'tmdb_id' in cleaned_data and cleaned_data['tmdb_id']:
                tmdb_id = cleaned_data['tmdb_id']
                if not str(tmdb_id).isdigit():
                    errors.append(f"Invalid TMDB ID format: {tmdb_id}")
                    cleaned_data['tmdb_id'] = ''
            
            # Validate image URL
            if 'image_url' in cleaned_data and cleaned_data['image_url']:
                image_url = cleaned_data['image_url']
                if not isinstance(image_url, str) or len(image_url) > 1000:
                    errors.append(f"Invalid image URL: {image_url}")
                    cleaned_data['image_url'] = ''
            
        except Exception as e:
            errors.append(f"Movie validation error: {str(e)}")
        
        return cleaned_data, errors
    
    @staticmethod
    def validate_genres(genres: Any) -> List[str]:
        """Validate and clean genre data"""
        if not genres:
            return []
        
        if isinstance(genres, str):
            # Split by comma and clean
            genre_list = [g.strip() for g in genres.split(',') if g.strip()]
        elif isinstance(genres, list):
            genre_list = [str(g).strip() for g in genres if g]
        else:
            return []
        
        # Filter out invalid genres
        valid_genres = []
        for genre in genre_list:
            if len(genre) <= 50 and re.match(r'^[a-zA-Z\s\-&]+$', genre):
                valid_genres.append(genre)
        
        return valid_genres[:10]  # Limit to 10 genres max

class ErrorHandler:
    """Utility class for consistent error handling"""
    
    @staticmethod
    def handle_scraping_error(error: Exception, context: str, logger) -> Dict[str, Any]:
        """Handle scraping errors consistently"""
        error_msg = f"Scraping error in {context}: {str(error)}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'error': error_msg,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def handle_db_error(error: Exception, operation: str, logger) -> Dict[str, Any]:
        """Handle database errors consistently"""
        error_msg = f"Database error during {operation}: {str(error)}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'error': error_msg,
            'operation': operation,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def create_success_response(data: Any, context: str = None) -> Dict[str, Any]:
        """Create consistent success response"""
        response = {
            'success': True,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            response['context'] = context
        
        return response