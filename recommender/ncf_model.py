# Neural Collaborative Filtering model using Keras
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def load_data(path="data/ratings.csv"):
    df = pd.read_csv(path)
    user_ids = df['userId'].unique()
    movie_ids = df['movieId'].unique()

    # Create mapping
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    df['user'] = df['userId'].map(user2user_encoded)
    df['movie'] = df['movieId'].map(movie2movie_encoded)

    num_users = len(user2user_encoded)
    num_movies = len(movie2movie_encoded)

    return df[['user', 'movie', 'rating']], num_users, num_movies, user2user_encoded, movie2movie_encoded

def build_ncf_model(num_users, num_movies, embedding_size=50):
    user_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, embedding_size)(user_input)
    user_vec = Flatten()(user_embedding)

    movie_input = Input(shape=(1,))
    movie_embedding = Embedding(num_movies, embedding_size)(movie_input)
    movie_vec = Flatten()(movie_embedding)

    concatenated = Concatenate()([user_vec, movie_vec])
    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1)(x)

    model = Model([user_input, movie_input], x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


from tensorflow.keras.models import load_model
import numpy as np

def load_trained_ncf_model(path="models/ncf_model.h5"):
    return load_model(path, compile=False)

def recommend_movies_ncf(model, user_id, movie_ids, user_map, movie_map, id_to_title, top_n=10):
    user_index = user_map.get(user_id)
    if user_index is None:
        return []

    # Filter only movie IDs present in mapping
    movie_indices = [movie_map[m] for m in movie_ids if m in movie_map]
    user_array = np.array([user_index] * len(movie_indices))
    movie_array = np.array(movie_indices)

    predictions = model.predict([user_array, movie_array], verbose=0)
    top_indices = predictions.flatten().argsort()[::-1][:top_n]

    top_movies = []
    for idx in top_indices:
        orig_movie_id = list(movie_map.keys())[list(movie_map.values()).index(movie_array[idx])]
        title = id_to_title.get(orig_movie_id, f"Movie ID {orig_movie_id}")
        score = float(predictions[idx])
        top_movies.append((title, score))

    return top_movies
