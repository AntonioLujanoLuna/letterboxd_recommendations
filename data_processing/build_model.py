#!/usr/local/bin/python3.11

import pandas as pd
import pickle

from surprise import Dataset
from surprise import Reader
from surprise import SVD

from surprise.model_selection import cross_validate
from surprise.dump import dump

import random
import numpy as np

def build_model(df, user_data, use_normalization=False):
    # print(df.head())

    # Set random seed so that returned recs are always the same for same user with same ratings
    # This might make sense so that results are consistent, or you might want to refresh with different results
    my_seed = 12
    random.seed(my_seed)
    np.random.seed(my_seed)
 
    user_rated = [x for x in user_data if x['rating_val'] > 0]

    user_df = pd.DataFrame(user_rated)
    df = pd.concat([df, user_df]).reset_index(drop=True)
    df.drop_duplicates(inplace=True)
    del user_df

    # Check if we should use normalization features
    if use_normalization:
        try:
            from rating_normalization import build_model_with_normalization
            return build_model_with_normalization(df, user_data)
        except ImportError:
            print("Rating normalization module not available, using standard approach")

    # Surprise dataset loading
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating_val"]], reader)
    del df

    # Configure algorithm
    algo = SVD()
    # cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

    trainingSet = data.build_full_trainset()
    algo.fit(trainingSet)

    user_watched_list = [x['movie_id'] for x in user_data]

    return algo, user_watched_list

if __name__ == "__main__":
    import os
    if os.getcwd().endswith("data_processing"):
        from get_user_ratings import get_user_data
    else:
        from data_processing.get_user_ratings import get_user_data

    # Load ratings data
    df = pd.read_csv('data/training_data.csv')

    user_data = get_user_data("samlearner")[0]
    algo, user_watched_list = build_model(df, user_data)

    dump("models/mini_model.pkl", predictions=None, algo=algo, verbose=1)
    with open("models/user_watched.txt", "wb") as fp:
        pickle.dump(user_watched_list, fp)
