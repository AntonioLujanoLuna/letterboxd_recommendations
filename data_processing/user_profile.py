#!/usr/local/bin/python3.11

"""
User profile building utilities for recommendation system
"""

import pandas as pd
from collections import Counter


def build_user_profile(user_data, movie_metadata_df=None):
    """
    Build a comprehensive user profile for better recommendations
    """
    if not user_data:
        return {}
    
    # Convert to dataframe
    user_df = pd.DataFrame([x for x in user_data if x['rating_val'] > 0])
    
    if user_df.empty:
        return {}
    
    # Merge with movie metadata if available
    if movie_metadata_df is not None:
        user_df = user_df.merge(movie_metadata_df, on='movie_id', how='left')
    
    # Calculate profile statistics
    profile = {
        'total_ratings': len(user_df),
        'average_rating': user_df['rating_val'].mean(),
        'rating_std': user_df['rating_val'].std(),
    }
    
    # Add year-based statistics if available
    if 'year_released' in user_df.columns:
        profile['average_year'] = user_df['year_released'].mean()
    else:
        profile['average_year'] = 2010  # Default
    
    # Add popularity statistics if available
    if 'count' in user_df.columns:
        profile['average_popularity'] = user_df['count'].mean()
    else:
        profile['average_popularity'] = 1000  # Default
    
    # Find favorite genres (from highly rated movies)
    if 'genres' in user_df.columns:
        highly_rated = user_df[user_df['rating_val'] >= 8]
        if not highly_rated.empty:
            all_genres = []
            for genres in highly_rated['genres'].dropna():
                if isinstance(genres, str):
                    all_genres.extend([g.strip() for g in genres.split(',')])
            
            if all_genres:
                genre_counts = Counter(all_genres)
                profile['favorite_genres'] = [g[0] for g in genre_counts.most_common(5)]
    
    if 'favorite_genres' not in profile:
        profile['favorite_genres'] = []
    
    # Rating distribution
    profile['rating_distribution'] = user_df['rating_val'].value_counts().to_dict()
    
    # Preference for obscure vs popular
    if 'count' in user_df.columns:
        profile['obscurity_preference'] = 1 / (user_df['count'].mean() / 1000)
    else:
        profile['obscurity_preference'] = 1.0
    
    return profile


def get_user_data_from_job_result(job_result):
    """
    Extract user data from a Redis job result
    """
    return job_result if isinstance(job_result, list) else [] 