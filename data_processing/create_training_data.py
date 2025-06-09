#!/usr/local/bin/python3.11

import pandas as pd

import numpy as np
from numpy import asarray
from numpy import savetxt

import pickle

from data_processing.run_model import create_popularity_buckets
import pymongo
import time
import random
from data_processing.db_connect import connect_to_db

def get_sample(collection, iteration_size, max_retries=5):
    """
    Get a sample from MongoDB with retry logic
    
    Args:
        collection: MongoDB collection (not a cursor!)
        iteration_size: Number of documents to sample
        max_retries: Maximum number of retry attempts
    """
    for attempt in range(max_retries):
        try:
            # Use aggregation pipeline for sampling
            rating_sample = collection.aggregate([{"$sample": {"size": iteration_size}}])
            return list(rating_sample)
        except pymongo.errors.OperationFailure as e:
            print(f"Encountered $sample operation error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Trying alternative sampling method...")
                # Fallback: use skip and limit with random offset
                total_docs = collection.estimated_document_count()
                if total_docs > iteration_size:
                    skip_amount = random.randint(0, max(0, total_docs - iteration_size))
                    return list(collection.find().skip(skip_amount).limit(iteration_size))
                else:
                    return list(collection.find().limit(iteration_size))
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return []

def create_training_data(db_client, sample_size=200000):
    ratings = db_client.ratings
    
    all_ratings = []
    seen_pairs = set()  # Track unique user-movie pairs
    
    while len(seen_pairs) < sample_size:
        rating_sample = get_sample(ratings, min(100000, sample_size - len(seen_pairs)))
        
        # Only add new unique ratings
        for rating in rating_sample:
            pair_key = (rating["user_id"], rating["movie_id"])
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_ratings.append(rating)
                
        if len(all_ratings) >= sample_size:
            break
            
        print(f"Unique records: {len(seen_pairs)}")
    
    # Convert to DataFrame only with needed data
    df = pd.DataFrame(all_ratings[:sample_size])
    df = df[["user_id", "movie_id", "rating_val"]]
    return df

def create_movie_data_sample(db_client, movie_list):
    movies = db_client.movies
    included_movies = movies.find({"movie_id": {"$in": movie_list}})

    movie_df = pd.DataFrame(list(included_movies))
    movie_df = movie_df[["movie_id", "image_url", "movie_title", "year_released"]]
    movie_df["image_url"] = (
        movie_df["image_url"]
        .fillna("")
        .str.replace("https://a.ltrbxd.com/resized/", "", regex=False)
    )
    movie_df["image_url"] = (
        movie_df["image_url"]
        .fillna("")
        .str.replace(
            "https://s.ltrbxd.com/static/img/empty-poster-230.c6baa486.png",
            "",
            regex=False,
        )
    )

    return movie_df


if __name__ == "__main__":
    # Connect to MongoDB client
    db_name, client, tmdb_key = connect_to_db()
    db = client[db_name]

    min_review_threshold = 20

    # Generate training data sample
    training_df = create_training_data(db, 1200000)

    # Create review counts dataframe
    review_count = db.ratings.aggregate(
        [
            {"$group": {"_id": "$movie_id", "review_count": {"$sum": 1}}},
            {"$match": {"review_count": {"$gte": min_review_threshold}}},
        ]
    )
    review_counts_df = pd.DataFrame(list(review_count))
    review_counts_df, popularity_tiers = create_popularity_buckets(review_counts_df)
    review_counts_df.rename(
        columns={"_id": "movie_id", "review_count": "count"}, inplace=True
    )

    threshold_movie_list = review_counts_df["movie_id"].to_list()

    # Generate movie data CSV
    movie_df = create_movie_data_sample(db, threshold_movie_list)
    print(movie_df.head())
    print(movie_df.shape)

    # Use movie_df to remove any items from threshold_list that do not have a "year_released"
    # This virtually always means it's a collection of more popular movies (such as the LOTR trilogy) and we don't want it included in recs
    retain_list = movie_df.loc[
        (movie_df["year_released"].notna() & movie_df["year_released"] != 0.0)
    ]["movie_id"].to_list()

    threshold_movie_list = [x for x in threshold_movie_list if x in retain_list]

    # Store Data
    with open("models/threshold_movie_list.txt", "wb") as fp:
        pickle.dump(threshold_movie_list, fp)

    training_df.to_csv("data/training_data.csv", index=False)
    review_counts_df.to_csv("data/review_counts.csv", index=False)
    movie_df.to_csv("../static/data/movie_data.csv", index=False)
    # Save the popularity tiers
    with open("models/popularity_tiers.pkl", "wb") as fp:
        pickle.dump(popularity_tiers, fp)
