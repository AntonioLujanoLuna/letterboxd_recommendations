#!/usr/local/bin/python3.11

from collections import defaultdict

from surprise import Dataset
from surprise import SVD
from surprise import Reader
from surprise.model_selection import GridSearchCV
from surprise import SVD
from surprise.dump import load

import pickle
import pandas as pd
import random

import os

import pymongo

try:
    from .db_config import config
except ImportError:
    config = None


def get_top_n(predictions, n=20):
    top_n = [(iid, est) for uid, iid, true_r, est, _ in predictions]
    top_n.sort(key=lambda x: (x[1], random.random()), reverse=True)

    return top_n[:n]


#!/usr/local/bin/python3.11

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle

def create_popularity_buckets(review_counts_df):
    """
    Create more intelligent popularity buckets using logarithmic scale
    """
    # Calculate log of review counts for better distribution
    review_counts_df['log_count'] = np.log10(review_counts_df['count'] + 1)
    
    # Create percentile-based buckets
    percentiles = [0, 50, 75, 90, 95, 99, 100]
    labels = ['obscure', 'indie', 'cult', 'popular', 'mainstream', 'blockbuster']
    
    review_counts_df['popularity_bucket'] = pd.qcut(
        review_counts_df['log_count'], 
        q=percentiles/100, 
        labels=labels,
        duplicates='drop'
    )
    
    # Also create decade-aware popularity
    # This would need movie year data, but here's the structure
    popularity_tiers = {
        'hidden_gems': review_counts_df[review_counts_df['count'] < 100]['movie_id'].tolist(),
        'undiscovered': review_counts_df[(review_counts_df['count'] >= 100) & (review_counts_df['count'] < 500)]['movie_id'].tolist(),
        'cult_favorites': review_counts_df[(review_counts_df['count'] >= 500) & (review_counts_df['count'] < 2000)]['movie_id'].tolist(),
        'well_known': review_counts_df[(review_counts_df['count'] >= 2000) & (review_counts_df['count'] < 10000)]['movie_id'].tolist(),
        'popular': review_counts_df[review_counts_df['count'] >= 10000]['movie_id'].tolist()
    }
    
    return review_counts_df, popularity_tiers

def get_recommendation_modes(predictions, movie_metadata, user_profile, watchlist=[]):
    """
    Create different recommendation modes from the same predictions
    """
    # Convert predictions to dataframe for easier manipulation
    pred_df = pd.DataFrame(predictions, columns=['movie_id', 'predicted_rating'])
    
    # Filter out watchlist items first
    if watchlist:
        pred_df = pred_df[~pred_df['movie_id'].isin(watchlist)]
    
    # Merge with movie metadata
    pred_df = pred_df.merge(movie_metadata, on='movie_id', how='left')
    
    # Calculate additional scores
    pred_df['diversity_score'] = calculate_diversity_score(pred_df, user_profile)
    pred_df['novelty_score'] = 1 / (np.log10(pred_df['count'] + 10))
    
    # Create different recommendation lists
    recommendations = {}
    
    # 1. Best Overall (classic approach)
    recommendations['best_overall'] = (
        pred_df.nlargest(50, 'predicted_rating')
        .to_dict('records')
    )
    
    # 2. Hidden Gems (high rating + low popularity)
    pred_df['hidden_gem_score'] = (
        pred_df['predicted_rating'] * 0.7 + 
        pred_df['novelty_score'] * 10 * 0.3
    )
    recommendations['hidden_gems'] = (
        pred_df[pred_df['count'] < 1000]
        .nlargest(50, 'hidden_gem_score')
        .to_dict('records')
    )
    
    # 3. Comfort Zone (high rating + similar to user taste)
    if user_profile and 'favorite_genres' in user_profile:
        pred_df['comfort_score'] = pred_df.apply(
            lambda x: calculate_genre_similarity(x, user_profile['favorite_genres']), 
            axis=1
        )
        pred_df['comfort_final'] = (
            pred_df['predicted_rating'] * 0.8 + 
            pred_df['comfort_score'] * 0.2
        )
        recommendations['comfort_zone'] = (
            pred_df.nlargest(50, 'comfort_final')
            .to_dict('records')
        )
    
    # 4. Expand Horizons (good rating + different from usual)
    pred_df['exploration_score'] = (
        pred_df['predicted_rating'] * 0.6 + 
        pred_df['diversity_score'] * 0.4
    )
    recommendations['expand_horizons'] = (
        pred_df[pred_df['predicted_rating'] > 7.0]
        .nlargest(50, 'exploration_score')
        .to_dict('records')
    )
    
    # 5. Critically Acclaimed (high rating + awards/recognition)
    # This would need additional data, but here's the structure
    if 'critic_score' in pred_df.columns:
        pred_df['critical_score'] = (
            pred_df['predicted_rating'] * 0.5 + 
            pred_df['critic_score'] * 0.5
        )
        recommendations['critically_acclaimed'] = (
            pred_df.nlargest(50, 'critical_score')
            .to_dict('records')
        )
    
    return recommendations

def calculate_diversity_score(movie_df, user_profile):
    """
    Calculate how different each movie is from user's typical preferences
    """
    if not user_profile or 'average_year' not in user_profile:
        return 1.0
    
    # Year diversity
    year_diff = abs(movie_df['year_released'] - user_profile['average_year']) / 50
    
    # Genre diversity (would need genre overlap calculation)
    # Popularity diversity
    pop_diversity = abs(movie_df['count'] - user_profile['average_popularity']) / 1000
    
    return (year_diff + pop_diversity) / 2

def calculate_genre_similarity(movie, favorite_genres):
    """
    Calculate genre overlap between movie and user preferences
    """
    if pd.isna(movie.get('genres')):
        return 0
    
    movie_genres = set(movie['genres'].split(',')) if isinstance(movie['genres'], str) else set()
    fav_genres = set(favorite_genres)
    
    if not movie_genres or not fav_genres:
        return 0
    
    intersection = len(movie_genres & fav_genres)
    union = len(movie_genres | fav_genres)
    
    return intersection / union if union > 0 else 0

def build_user_profile(user_data, movie_metadata):
    """
    Build a comprehensive user profile for better recommendations
    """
    if not user_data:
        return {}
    
    # Convert to dataframe
    user_df = pd.DataFrame([x for x in user_data if x['rating_val'] > 0])
    
    if user_df.empty:
        return {}
    
    # Merge with movie metadata
    user_df = user_df.merge(movie_metadata, on='movie_id', how='left')
    
    # Calculate profile statistics
    profile = {
        'total_ratings': len(user_df),
        'average_rating': user_df['rating_val'].mean(),
        'rating_std': user_df['rating_val'].std(),
        'average_year': user_df['year_released'].mean() if 'year_released' in user_df else 2000,
        'average_popularity': user_df['count'].mean() if 'count' in user_df else 1000,
    }
    
    # Find favorite genres (from highly rated movies)
    highly_rated = user_df[user_df['rating_val'] >= 8]
    if not highly_rated.empty and 'genres' in highly_rated.columns:
        all_genres = []
        for genres in highly_rated['genres'].dropna():
            if isinstance(genres, str):
                all_genres.extend(genres.split(','))
        
        from collections import Counter
        genre_counts = Counter(all_genres)
        profile['favorite_genres'] = [g[0] for g in genre_counts.most_common(5)]
    
    # Rating distribution
    profile['rating_distribution'] = user_df['rating_val'].value_counts().to_dict()
    
    # Preference for obscure vs popular
    profile['obscurity_preference'] = 1 / (user_df['count'].mean() / 1000) if 'count' in user_df else 1.0
    
    return profile

def run_model(username, algo, user_watched_list, threshold_movie_list, 
                       movie_metadata_df, user_watchlist=[], num_recommendations=50,
                       recommendation_mode='best_overall'):
    """
    Enhanced model with multiple recommendation modes and better filtering
    """
    # Filter out watched movies and watchlist
    all_excluded = set(user_watched_list) | set(user_watchlist)
    unwatched_movies = [x for x in threshold_movie_list if x not in all_excluded]
    
    # Create prediction set
    prediction_set = [(username, x, 0) for x in unwatched_movies]
    
    # Get predictions
    predictions = algo.test(prediction_set)
    
    # Extract predictions with denormalization if needed
    if hasattr(algo, 'user_norm_params'):
        # Denormalize predictions back to 1-10 scale
        mean = algo.user_norm_params['mean']
        std = algo.user_norm_params['std']
        
        top_n = []
        for uid, iid, true_r, est, _ in predictions:
            # Reverse normalization
            denormalized = (est * std) + mean
            # Clip to valid range
            denormalized = max(1, min(10, denormalized))
            top_n.append((iid, denormalized))
    else:
        top_n = [(iid, est) for uid, iid, true_r, est, _ in predictions]
    
    # Sort by rating
    top_n.sort(key=lambda x: x[1], reverse=True)
    
    # Build user profile
    from handle_recs import get_client_user_data
    user_data = get_client_user_data(username, False)
    user_profile = build_user_profile(user_data, movie_metadata_df)
    
    # Get recommendations in different modes
    all_recommendations = get_recommendation_modes(
        top_n[:num_recommendations * 3],  # Get extra for filtering
        movie_metadata_df,
        user_profile,
        user_watchlist
    )
    
    # Return requested mode
    return all_recommendations.get(recommendation_mode, all_recommendations['best_overall'])

if __name__ == "__main__":
    with open("models/user_watched.txt", "rb") as fp:
        user_watched_list = pickle.load(fp)

    with open("models/threshold_movie_list.txt", "rb") as fp:
        threshold_movie_list = pickle.load(fp)

    algo = load("models/mini_model.pkl")[1]

    recs = run_model("antonio_lujano", algo, user_watched_list, threshold_movie_list, 25)
    print(recs)