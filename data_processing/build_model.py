# Replace the build_model function in data_processing/build_model.py

import pandas as pd
import pickle
import gc
from typing import List, Dict, Any, Tuple

from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from surprise.dump import dump

import random
import numpy as np

from config import config

# Setup logging
try:
    from .utils.logging_config import setup_logger
    logger = setup_logger('letterboxd.model')
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def build_model(df: pd.DataFrame, user_data: List[Dict[str, Any]], 
                use_normalization: bool = False) -> Tuple[Any, List[str]]:
    """
    Build SVD recommendation model with improved configuration and error handling
    
    Args:
        df: Training dataframe with columns [user_id, movie_id, rating_val]
        user_data: List of user rating dictionaries
        use_normalization: Whether to use rating normalization
    
    Returns:
        Tuple of (trained_algorithm, user_watched_list)
    """
    try:
        logger.info(f"Building model with {len(df)} training samples and {len(user_data)} user ratings")
        
        # Set random seed for reproducible results
        my_seed = 12
        random.seed(my_seed)
        np.random.seed(my_seed)
        
        # Filter user data for rated movies only
        user_rated = [x for x in user_data if x.get('rating_val', 0) > 0]
        logger.info(f"User has {len(user_rated)} rated movies")
        
        if not user_rated:
            raise ValueError("User has no rated movies")
        
        # Create user dataframe and combine with training data
        user_df = pd.DataFrame(user_rated)
        
        # Validate user data columns
        required_columns = ['user_id', 'movie_id', 'rating_val']
        missing_columns = [col for col in required_columns if col not in user_df.columns]
        if missing_columns:
            raise ValueError(f"User data missing required columns: {missing_columns}")
        
        # Combine datasets
        df = pd.concat([df, user_df]).reset_index(drop=True)
        df.drop_duplicates(subset=['user_id', 'movie_id'], inplace=True)
        
        logger.info(f"Combined dataset size: {len(df)} ratings")
        
        # Clean up user_df immediately
        del user_df
        gc.collect()
        
        # Check if we should use normalization features
        if use_normalization:
            try:
                from .rating_normalization import build_model_with_normalization
                logger.info("Using rating normalization")
                algo, user_watched_list, user_stats = build_model_with_normalization(df, user_data)
                # Store user_stats in algo for later use if needed
                algo.user_stats = user_stats
                return algo, user_watched_list
            except ImportError:
                logger.warning("Rating normalization module not available, using standard approach")
        
        # Validate rating scale
        min_rating = df['rating_val'].min()
        max_rating = df['rating_val'].max()
        logger.info(f"Rating scale: {min_rating} to {max_rating}")
        
        if min_rating < 1 or max_rating > 10:
            logger.warning(f"Unusual rating scale detected: {min_rating}-{max_rating}")
        
        # Create Surprise dataset
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(df[required_columns], reader)
        
        # Clean up dataframe to save memory
        df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"Training dataframe size: {df_size_mb:.1f} MB")
        del df
        gc.collect()
        
        # Configure SVD algorithm with parameters from config
        algo = SVD(
            n_factors=config.model.svd_factors,
            n_epochs=config.model.svd_epochs,
            lr_all=config.model.svd_learning_rate,
            reg_all=config.model.svd_regularization,
            random_state=my_seed,
            verbose=True
        )
        
        logger.info(f"SVD configuration: factors={config.model.svd_factors}, epochs={config.model.svd_epochs}")
        
        # Optional: Run cross-validation for model evaluation
        if len(user_rated) > 10:  # Only if user has enough ratings
            try:
                logger.info("Running cross-validation...")
                cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
                logger.info(f"CV RMSE: {cv_results['test_rmse'].mean():.3f} (+/- {cv_results['test_rmse'].std() * 2:.3f})")
                logger.info(f"CV MAE: {cv_results['test_mae'].mean():.3f} (+/- {cv_results['test_mae'].std() * 2:.3f})")
            except Exception as e:
                logger.warning(f"Cross-validation failed: {str(e)}")
        
        # Build full training set and fit model
        logger.info("Training SVD model...")
        training_set = data.build_full_trainset()
        algo.fit(training_set)
        
        # Create user watched list
        user_watched_list = [x['movie_id'] for x in user_data if 'movie_id' in x]
        logger.info(f"User has watched {len(user_watched_list)} movies total")
        
        # Log model statistics
        n_users = training_set.n_users
        n_items = training_set.n_items
        n_ratings = training_set.n_ratings
        sparsity = 1 - (n_ratings / (n_users * n_items))
        
        logger.info(f"Model statistics:")
        logger.info(f"  Users: {n_users}")
        logger.info(f"  Movies: {n_items}")
        logger.info(f"  Ratings: {n_ratings}")
        logger.info(f"  Sparsity: {sparsity:.4f}")
        
        return algo, user_watched_list
        
    except Exception as e:
        logger.error(f"Model building failed: {str(e)}")
        raise ValueError(f"Failed to build recommendation model: {str(e)}")

def save_model(algo, user_watched_list: List[str], model_path: str = "models/mini_model.pkl", 
               watchlist_path: str = "models/user_watched.txt") -> None:
    """Save trained model and user watchlist"""
    try:
        # Save model
        dump(model_path, predictions=None, algo=algo, verbose=1)
        logger.info(f"Model saved to {model_path}")
        
        # Save user watched list
        with open(watchlist_path, "wb") as fp:
            pickle.dump(user_watched_list, fp)
        logger.info(f"User watched list saved to {watchlist_path}")
        
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise

def load_model(model_path: str = "models/mini_model.pkl", 
               watchlist_path: str = "models/user_watched.txt") -> Tuple[Any, List[str]]:
    """Load trained model and user watchlist"""
    try:
        from surprise.dump import load
        
        # Load model
        _, algo = load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load user watched list
        with open(watchlist_path, "rb") as fp:
            user_watched_list = pickle.load(fp)
        logger.info(f"User watched list loaded from {watchlist_path}")
        
        return algo, user_watched_list
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

if __name__ == "__main__":
    from data_processing.get_user_ratings import get_user_data
    
    # Load ratings data
    df = pd.read_csv('data/training_data.csv')
    logger.info(f"Loaded training data: {len(df)} ratings")

    user_data = get_user_data("samlearner")[0]
    logger.info(f"Loaded user data: {len(user_data)} ratings")
    
    # Build model
    algo, user_watched_list = build_model(df, user_data)
    
    # Save model
    save_model(algo, user_watched_list)