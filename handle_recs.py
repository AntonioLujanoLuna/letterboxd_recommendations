import pandas as pd
import pickle
from typing import List, Dict, Any, Optional

from rq import Queue, get_current_job
from rq.job import Job
from rq.registry import FinishedJobRegistry, DeferredJobRegistry

from data_processing.get_user_ratings import get_user_data
from data_processing.get_user_watchlist import get_watchlist_data
from data_processing.build_model import build_model
from data_processing.run_model import run_model

from worker import conn
from config import Config
from utils.logging_config import setup_logger
from utils.validation import DataValidator, ErrorHandler
from utils.validation import ValidationError

logger = setup_logger('letterboxd.recommendations')

def get_previous_job_from_registry(index: int = -1) -> Optional[Job]:
    """Get previous job from registry with error handling"""
    try:
        q = Queue('high', connection=conn)
        registry = FinishedJobRegistry(queue=q)
        job_ids = registry.get_job_ids()
        
        if not job_ids:
            logger.warning("No jobs found in registry")
            return None
        
        job_id = job_ids[index]
        job = q.fetch_job(job_id)
        return job
        
    except Exception as e:
        logger.error(f"Error retrieving job from registry: {str(e)}")
        return None

def filter_threshold_list(threshold_movie_list: List[str], 
                         review_count_threshold: int = 2000) -> List[str]:
    """Filter movie list by review count threshold with error handling"""
    try:
        review_counts = pd.read_csv('data_processing/data/review_counts.csv')
        
        if review_counts.empty:
            logger.warning("Review counts file is empty")
            return threshold_movie_list
        
        filtered_counts = review_counts.loc[review_counts['count'] < review_count_threshold]
        included_movies = filtered_counts['movie_id'].to_list()
        
        filtered_list = [x for x in threshold_movie_list if x in included_movies]
        
        logger.info(f"Filtered movies from {len(threshold_movie_list)} to {len(filtered_list)} based on review threshold {review_count_threshold}")
        
        return filtered_list
        
    except FileNotFoundError:
        logger.error("Review counts file not found, returning original list")
        return threshold_movie_list
    except Exception as e:
        logger.error(f"Error filtering threshold list: {str(e)}")
        return threshold_movie_list

def get_client_user_data(username: str, data_opt_in: bool) -> List[Dict[str, Any]]:
    """Get and validate user data with improved error handling"""
    current_job = get_current_job(conn)
    
    try:
        # Validate username
        if not Config.validate_username(username):
            raise ValidationError(f"Invalid username: {username}")
        
        logger.info(f"Fetching user data for: {username}")
        
        # Get user ratings
        user_data_result = get_user_data(username, data_opt_in)
        
        if not user_data_result or len(user_data_result) != 2:
            raise ValueError("Invalid user data response format")
        
        user_data, user_status = user_data_result
        
        # Validate user data
        if user_status != "success":
            logger.warning(f"User data fetch failed with status: {user_status}")
            current_job.meta['user_status'] = user_status
            current_job.meta['num_user_ratings'] = 0
            current_job.meta['user_watchlist'] = []
            current_job.save()
            return []
        
        # Validate and clean user data
        validated_data, validation_errors = DataValidator.validate_user_data(user_data)
        
        if validation_errors:
            logger.warning(f"User data validation errors for {username}: {validation_errors}")
        
        logger.info(f"Retrieved {len(validated_data)} valid ratings for {username}")
        
        # Get user watchlist
        try:
            watchlist_result = get_watchlist_data(username)
            user_watchlist = watchlist_result[0] if watchlist_result and watchlist_result[1] == "success" else []
        except Exception as e:
            logger.warning(f"Failed to get watchlist for {username}: {str(e)}")
            user_watchlist = []
        
        # Update job metadata
        current_job.meta['user_status'] = user_status
        current_job.meta['num_user_ratings'] = len(validated_data)
        current_job.meta['user_watchlist'] = user_watchlist
        current_job.save()
        
        return validated_data
        
    except ValidationError:
        raise
    except Exception as e:
        error_response = ErrorHandler.handle_scraping_error(e, f"get_client_user_data({username})", logger)
        
        # Update job with error
        current_job.meta['user_status'] = "error"
        current_job.meta['error'] = error_response
        current_job.save()
        
        return []

def build_client_model(username: str, 
                      training_data_rows: int = 500000, 
                      popularity_threshold: Optional[int] = None, 
                      num_items: int = 50) -> List[Dict[str, Any]]:
    """Build recommendation model with improved error handling"""
    current_job = get_current_job(conn)
    
    try:
        # Validate inputs
        if training_data_rows <= 0 or training_data_rows > Config.MAX_TRAINING_SIZE:
            raise ValidationError(f"Invalid training data size: {training_data_rows}")
        
        if not Config.validate_username(username):
            raise ValidationError(f"Invalid username: {username}")
        
        logger.info(f"Building model for {username} with {training_data_rows} samples")
        
        # Load user data from previous Redis job
        user_data_job = current_job.dependency
        if not user_data_job:
            raise ValueError("No user data job dependency found")
        
        user_data = user_data_job.result
        if not user_data:
            raise ValueError("No user data available from previous job")
        
        # Update job stage
        current_job.meta['stage'] = 'loading_training_data'
        current_job.save()
        
        # Load training dataset
        try:
            df = pd.read_csv('data_processing/data/training_data.csv')
            logger.info(f"Loaded training data with {len(df)} samples")
        except FileNotFoundError:
            raise ValueError("Training data file not found")
        
        # Sample data intelligently if needed
        if training_data_rows < len(df):
            try:
                from data_processing.rating_normalization import intelligent_sampling
                model_df = intelligent_sampling(df, training_data_rows)
                logger.info(f"Applied intelligent sampling: {len(model_df)} samples")
            except ImportError:
                model_df = df.head(training_data_rows)
                logger.info(f"Applied simple sampling: {len(model_df)} samples")
        else:
            model_df = df
        
        # Load threshold movie list
        current_job.meta['stage'] = 'loading_movie_data'
        current_job.save()
        
        try:
            with open("data_processing/models/threshold_movie_list.txt", "rb") as fp:
                threshold_movie_list = pickle.load(fp)
            logger.info(f"Loaded {len(threshold_movie_list)} threshold movies")
        except FileNotFoundError:
            raise ValueError("Threshold movie list file not found")
        
        # Load movie metadata
        try:
            movie_metadata_df = pd.read_csv('static/data/movie_data.csv')
            logger.info(f"Loaded metadata for {len(movie_metadata_df)} movies")
            
            # Merge with review counts
            try:
                review_counts_df = pd.read_csv('data_processing/data/review_counts.csv')
                movie_metadata_df = movie_metadata_df.merge(
                    review_counts_df[['movie_id', 'count']], 
                    on='movie_id', 
                    how='left'
                )
                movie_metadata_df['count'] = movie_metadata_df['count'].fillna(100)
            except FileNotFoundError:
                logger.warning("Review counts file not found, using default counts")
                movie_metadata_df['count'] = 100
                
        except FileNotFoundError:
            raise ValueError("Movie metadata file not found")
        
        # Apply popularity filter if requested
        if popularity_threshold:
            threshold_movie_list = filter_threshold_list(threshold_movie_list, popularity_threshold)
            logger.info(f"Applied popularity filter: {len(threshold_movie_list)} movies remaining")
        
        # Build model
        current_job.meta['stage'] = 'building_model'
        current_job.save()
        
        try:
            algo, user_watched_list = build_model(model_df, user_data)
            logger.info(f"Model built successfully, user has watched {len(user_watched_list)} movies")
        except Exception as e:
            raise ValueError(f"Model building failed: {str(e)}")
        
        # Clean up memory
        del model_df
        
        # Run model for recommendations
        current_job.meta['stage'] = 'generating_recommendations'
        current_job.save()
        
        try:
            user_watchlist = current_job.meta.get('user_watchlist', [])
            recs = run_model(
                username, 
                algo, 
                user_watched_list, 
                threshold_movie_list, 
                movie_metadata_df, 
                user_watchlist, 
                num_items, 
                'best_overall', 
                user_data
            )
            
            logger.info(f"Generated {len(recs)} recommendations for {username}")
            
            current_job.meta['stage'] = 'completed'
            current_job.save()
            
            return recs
            
        except Exception as e:
            raise ValueError(f"Recommendation generation failed: {str(e)}")
        
    except (ValidationError, ValueError):
        raise
    except Exception as e:
        error_response = ErrorHandler.handle_scraping_error(
            e, f"build_client_model({username})", logger
        )
        
        current_job.meta['stage'] = 'error'
        current_job.meta['error'] = error_response
        current_job.save()
        
        raise ValueError(f"Model building failed: {str(e)}")

def get_job_statistics() -> Dict[str, Any]:
    """Get statistics about current jobs"""
    try:
        stats = {}
        for queue_name in Config.REDIS_QUEUES:
            queue = Queue(queue_name, connection=conn)
            stats[queue_name] = {
                'queued': len(queue),
                'deferred': DeferredJobRegistry(queue=queue).count,
                'failed': queue.failed_job_registry.count
            }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting job statistics: {str(e)}")
        return {}

def cleanup_old_jobs(max_age_hours: int = 24):
    """Clean up old finished jobs"""
    try:
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        cleaned_count = 0
        for queue_name in Config.REDIS_QUEUES:
            queue = Queue(queue_name, connection=conn)
            registry = FinishedJobRegistry(queue=queue)
            
            for job_id in registry.get_job_ids():
                try:
                    job = queue.fetch_job(job_id)
                    if job and job.ended_at and job.ended_at < cutoff_time:
                        job.delete()
                        cleaned_count += 1
                except:
                    continue
        
        logger.info(f"Cleaned up {cleaned_count} old jobs")
        return cleaned_count
        
    except Exception as e:
        logger.error(f"Error cleaning up jobs: {str(e)}")
        return 0