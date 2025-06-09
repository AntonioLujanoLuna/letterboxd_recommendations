# Replace the build_client_model function in handle_recs.py with this improved version:

import pandas as pd
import pickle
import gc
from typing import List, Dict, Any, Optional

from rq import get_current_job
from rq.job import Job

from data_processing.get_user_ratings import get_user_data
from data_processing.get_user_watchlist import get_watchlist_data
from data_processing.build_model import build_model
from data_processing.run_model import run_model

from worker import conn
from config import config

try:
    from data_processing.utils.logging_config import setup_logger
    from data_processing.utils.validation import DataValidator, ErrorHandler, ValidationError
    logger = setup_logger('letterboxd.recommendations')
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class MemoryManager:
    """Context manager for memory cleanup"""
    
    def __init__(self, *variables):
        self.variables = variables
        self.initial_memory = None
    
    def __enter__(self):
        try:
            import psutil
            process = psutil.Process()
            self.initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Initial memory usage: {self.initial_memory:.1f} MB")
        except ImportError:
            pass
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Force cleanup of variables
        for var_name in self.variables:
            if var_name in locals() or var_name in globals():
                try:
                    if var_name in locals():
                        del locals()[var_name]
                    if var_name in globals():
                        del globals()[var_name]
                except:
                    pass
        
        # Force garbage collection
        gc.collect()
        
        try:
            import psutil
            process = psutil.Process()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            if self.initial_memory:
                logger.info(f"Final memory usage: {final_memory:.1f} MB (freed: {self.initial_memory - final_memory:.1f} MB)")
        except ImportError:
            pass

def intelligent_sampling(df: pd.DataFrame, target_size: int) -> pd.DataFrame:
    """Intelligent sampling to maintain diversity while reducing memory"""
    if len(df) <= target_size:
        return df
    
    logger.info(f"Applying intelligent sampling: {len(df)} -> {target_size} ratings")
    
    # Calculate user rating counts
    user_counts = df['user_id'].value_counts()
    
    # Categorize users by activity level
    power_users = user_counts[user_counts > 500].index
    regular_users = user_counts[(user_counts >= 50) & (user_counts <= 500)].index
    casual_users = user_counts[user_counts < 50].index
    
    samples = []
    
    # 20% from power users (but cap their individual contributions)
    if len(power_users) > 0:
        power_df = df[df['user_id'].isin(power_users)]
        power_sample = power_df.groupby('user_id').apply(
            lambda x: x.sample(min(len(x), 200))  # Cap at 200 ratings per power user
        ).reset_index(drop=True)
        samples.append(power_sample.sample(min(len(power_sample), int(target_size * 0.2))))
        del power_df, power_sample
    
    # 60% from regular users
    if len(regular_users) > 0:
        regular_df = df[df['user_id'].isin(regular_users)]
        samples.append(regular_df.sample(min(len(regular_df), int(target_size * 0.6))))
        del regular_df
    
    # 20% from casual users (preserve all their ratings)
    if len(casual_users) > 0:
        casual_df = df[df['user_id'].isin(casual_users)]
        samples.append(casual_df.sample(min(len(casual_df), int(target_size * 0.2))))
        del casual_df
    
    # Combine samples
    if samples:
        final_sample = pd.concat(samples, ignore_index=True)
        del samples  # Clear the list
        
        # If still over target, randomly sample down
        if len(final_sample) > target_size:
            final_sample = final_sample.sample(target_size)
        
        logger.info(f"Intelligent sampling completed: {len(final_sample)} ratings selected")
        return final_sample
    else:
        # Fallback to simple random sampling
        return df.sample(min(target_size, len(df)))

def load_data_with_memory_management(training_data_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load and prepare data with memory management"""
    
    # Load training dataset
    try:
        logger.info("Loading training data...")
        df = pd.read_csv('data_processing/data/training_data.csv')
        logger.info(f"Loaded training data with {len(df)} samples, {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    except FileNotFoundError:
        raise ValueError("Training data file not found")
    
    # Apply intelligent sampling if needed
    if training_data_rows < len(df):
        df = intelligent_sampling(df, training_data_rows)
        gc.collect()  # Clean up after sampling
    
    # Load movie metadata
    try:
        logger.info("Loading movie metadata...")
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
            del review_counts_df  # Clean up immediately
        except FileNotFoundError:
            logger.warning("Review counts file not found, using default counts")
            movie_metadata_df['count'] = 100
            
    except FileNotFoundError:
        raise ValueError("Movie metadata file not found")
    
    # Load threshold movie list
    try:
        with open("data_processing/models/threshold_movie_list.txt", "rb") as fp:
            threshold_movie_list = pickle.load(fp)
        logger.info(f"Loaded {len(threshold_movie_list)} threshold movies")
    except FileNotFoundError:
        raise ValueError("Threshold movie list file not found")
    
    return df, movie_metadata_df, threshold_movie_list

def build_client_model(username: str, 
                      training_data_rows: int = 500000, 
                      popularity_threshold: Optional[int] = None, 
                      num_items: int = 50) -> List[Dict[str, Any]]:
    """Build recommendation model with improved error handling and memory management"""
    current_job = get_current_job(conn)
    
    # Use memory manager for the entire operation
    with MemoryManager('df', 'model_df', 'movie_metadata_df'):
        try:
            # Validate inputs
            if training_data_rows <= 0 or training_data_rows > config.model.max_training_size:
                raise ValidationError(f"Invalid training data size: {training_data_rows}")
            
            if not config.validate_username(username):
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
            
            # Load and prepare data
            df, movie_metadata_df, threshold_movie_list = load_data_with_memory_management(training_data_rows)
            
            # Apply popularity filter if requested
            if popularity_threshold:
                threshold_movie_list = filter_threshold_list(threshold_movie_list, popularity_threshold)
                logger.info(f"Applied popularity filter: {len(threshold_movie_list)} movies remaining")
            
            # Build model
            current_job.meta['stage'] = 'building_model'
            current_job.save()
            
            try:
                algo, user_watched_list = build_model(df, user_data)
                logger.info(f"Model built successfully, user has watched {len(user_watched_list)} movies")
            except Exception as e:
                raise ValueError(f"Model building failed: {str(e)}")
            
            # Clean up training data immediately after model building
            del df
            gc.collect()
            
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
            error_msg = f"Model building failed: {str(e)}"
            logger.error(error_msg)
            
            current_job.meta['stage'] = 'error'
            current_job.meta['error'] = error_msg
            current_job.save()
            
            raise ValueError(error_msg)

def filter_threshold_list(threshold_movie_list: List[str], 
                         review_count_threshold: int = 2000) -> List[str]:
    """Filter movie list by review count threshold with improved error handling"""
    try:
        review_counts = pd.read_csv('data_processing/data/review_counts.csv')
        
        if review_counts.empty:
            logger.warning("Review counts file is empty")
            return threshold_movie_list
        
        filtered_counts = review_counts.loc[review_counts['count'] < review_count_threshold]
        included_movies = filtered_counts['movie_id'].to_list()
        
        filtered_list = [x for x in threshold_movie_list if x in included_movies]
        
        logger.info(f"Filtered movies from {len(threshold_movie_list)} to {len(filtered_list)} based on review threshold {review_count_threshold}")
        
        # Clean up
        del review_counts, filtered_counts
        
        return filtered_list
        
    except FileNotFoundError:
        logger.error("Review counts file not found, returning original list")
        return threshold_movie_list
    except Exception as e:
        logger.error(f"Error filtering threshold list: {str(e)}")
        return threshold_movie_list

# Also update get_client_user_data with better validation
def get_client_user_data(username: str, data_opt_in: bool) -> List[Dict[str, Any]]:
    """Get and validate user data with improved error handling"""
    current_job = get_current_job(conn)
    
    try:
        # Validate username
        if not config.validate_username(username.lower().strip()):
            raise ValidationError(f"Invalid username: {username}")
        
        username = username.lower().strip()
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
        try:
            validated_data, validation_errors = DataValidator.validate_user_data(user_data)
            
            if validation_errors:
                logger.warning(f"User data validation errors for {username}: {validation_errors[:5]}")  # Log first 5 errors
        except:
            # Fallback validation
            validated_data = [r for r in user_data if isinstance(r, dict) and 
                            'movie_id' in r and 'rating_val' in r and 'user_id' in r]
        
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
        error_msg = f"Error fetching user data for {username}: {str(e)}"
        logger.error(error_msg)
        
        # Update job with error
        current_job.meta['user_status'] = "error"
        current_job.meta['error'] = error_msg
        current_job.save()
        
        return []