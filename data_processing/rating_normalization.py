#!/usr/local/bin/python3.11

import pandas as pd
import numpy as np
from collections import defaultdict

def normalize_user_ratings(df):
    """
    Normalize ratings to account for user bias
    Some users rate everything high, others are harsh critics
    """
    # Calculate user statistics
    user_stats = df.groupby('user_id')['rating_val'].agg([
        'mean', 
        'std', 
        'count'
    ]).reset_index()
    
    # Handle users with very few ratings or no variance
    user_stats['std'] = user_stats['std'].fillna(1.0)
    user_stats.loc[user_stats['std'] < 0.5, 'std'] = 1.0
    
    # Create a mapping dictionary for faster lookup
    user_mean_map = dict(zip(user_stats['user_id'], user_stats['mean']))
    user_std_map = dict(zip(user_stats['user_id'], user_stats['std']))
    
    # Normalize ratings using z-score normalization
    df['normalized_rating'] = df.apply(
        lambda row: (row['rating_val'] - user_mean_map[row['user_id']]) / user_std_map[row['user_id']], 
        axis=1
    )
    
    # Also create a 0-1 normalized version
    df['min_max_normalized'] = df.apply(
        lambda row: (row['rating_val'] - 1) / 9,  # Assuming 1-10 scale
        axis=1
    )
    
    # Blend both normalizations (z-score can be too aggressive)
    df['final_normalized_rating'] = (
        0.7 * df['normalized_rating'] + 
        0.3 * df['min_max_normalized']
    )
    
    # Clip extreme values
    df['final_normalized_rating'] = df['final_normalized_rating'].clip(-3, 3)
    
    return df, user_stats

def apply_time_decay(df, decay_factor=0.95):
    """
    Apply time decay to ratings - recent ratings are weighted more heavily
    """
    if 'rating_date' not in df.columns:
        # If no date info, return unchanged
        return df
    
    # Convert to datetime if needed
    df['rating_date'] = pd.to_datetime(df['rating_date'])
    
    # Calculate days since rating
    latest_date = df['rating_date'].max()
    df['days_ago'] = (latest_date - df['rating_date']).dt.days
    
    # Apply exponential decay
    # Ratings lose (1-decay_factor) of their weight per year
    df['time_weight'] = decay_factor ** (df['days_ago'] / 365)
    
    # Apply weight to normalized ratings
    df['weighted_rating'] = df['final_normalized_rating'] * df['time_weight']
    
    return df

def intelligent_sampling(df, target_size=500000):
    """
    Sample ratings intelligently to maintain diversity
    """
    if len(df) <= target_size:
        return df
    
    # Calculate user rating counts
    user_counts = df['user_id'].value_counts()
    
    # Categorize users
    power_users = user_counts[user_counts > 500].index
    regular_users = user_counts[(user_counts >= 50) & (user_counts <= 500)].index
    casual_users = user_counts[user_counts < 50].index
    
    # Sample proportionally
    samples = []
    
    # 20% from power users (but cap their individual contributions)
    power_df = df[df['user_id'].isin(power_users)]
    power_sample = power_df.groupby('user_id').apply(
        lambda x: x.sample(min(len(x), 200))  # Cap at 200 ratings per power user
    ).reset_index(drop=True)
    samples.append(power_sample.sample(min(len(power_sample), int(target_size * 0.2))))
    
    # 60% from regular users
    regular_df = df[df['user_id'].isin(regular_users)]
    samples.append(regular_df.sample(min(len(regular_df), int(target_size * 0.6))))
    
    # 20% from casual users (preserve all their ratings)
    casual_df = df[df['user_id'].isin(casual_users)]
    samples.append(casual_df.sample(min(len(casual_df), int(target_size * 0.2))))
    
    # Combine samples
    final_sample = pd.concat(samples, ignore_index=True)
    
    # If still over target, randomly sample down
    if len(final_sample) > target_size:
        final_sample = final_sample.sample(target_size)
    
    return final_sample

# Modified build_model.py integration
def build_model_with_normalization(df, user_data):
    """
    Enhanced build_model function with normalization
    """
    import random
    import numpy as np
    from surprise import Dataset, Reader, SVD
    
    # Set random seed
    my_seed = 12
    random.seed(my_seed)
    np.random.seed(my_seed)
    
    # Prepare user data
    user_rated = [x for x in user_data if x['rating_val'] > 0]
    user_df = pd.DataFrame(user_rated)
    
    # Combine with training data
    df = pd.concat([df, user_df]).reset_index(drop=True)
    df.drop_duplicates(inplace=True)
    
    # Apply normalization
    print("Normalizing ratings...")
    df, user_stats = normalize_user_ratings(df)
    
    # Apply intelligent sampling if needed
    if len(df) > 500000:
        print(f"Sampling from {len(df)} to 500000 ratings...")
        df = intelligent_sampling(df, 500000)
    
    # Use normalized ratings for training
    reader = Reader(rating_scale=(-3, 3))  # Adjusted for normalized scale
    data = Dataset.load_from_df(
        df[["user_id", "movie_id", "final_normalized_rating"]], 
        reader
    )
    
    # Configure and train algorithm
    algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    trainingSet = data.build_full_trainset()
    algo.fit(trainingSet)
    
    # Store normalization parameters for the target user
    target_user_stats = user_stats[user_stats['user_id'] == user_data[0]['user_id']]
    if len(target_user_stats) > 0:
        user_mean = target_user_stats.iloc[0]['mean']
        user_std = target_user_stats.iloc[0]['std']
    else:
        # Calculate for new user
        user_ratings = [x['rating_val'] for x in user_rated]
        user_mean = np.mean(user_ratings)
        user_std = np.std(user_ratings) if len(user_ratings) > 1 else 1.0
    
    # Store for denormalization later
    algo.user_norm_params = {'mean': user_mean, 'std': user_std}
    
    user_watched_list = [x['movie_id'] for x in user_data]
    
    return algo, user_watched_list, user_stats