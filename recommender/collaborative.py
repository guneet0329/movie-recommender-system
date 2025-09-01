# Collaborative Filtering model using SVD or ALS
from surprise import SVD, Dataset, Reader, dump
from surprise.model_selection import train_test_split
import pandas as pd
import os

MODEL_PATH = "models/svd_model.pkl"

def train_svd_model(ratings_csv, save_model=True):
    df = pd.read_csv(ratings_csv)[['userId', 'movieId', 'rating']]
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df, reader)
    trainset = data.build_full_trainset()

    algo = SVD()
    algo.fit(trainset)

    if save_model:
        dump.dump(MODEL_PATH, algo=algo)

    return algo

def load_trained_model():
    if os.path.exists(MODEL_PATH):
        _, algo = dump.load(MODEL_PATH)
        return algo
    else:
        raise FileNotFoundError("Model not found. Train it first.")


def recommend_movies(algo, user_id, movie_ids, n=10):
    # Recommend top n movies for a given user
    predictions = [algo.predict(user_id, movie_id) for movie_id in movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]
    
    return [(pred.iid, pred.est) for pred in top_n]
