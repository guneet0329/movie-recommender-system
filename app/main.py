import os
import sys

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import pandas as pd
import streamlit as st
from recommender.collaborative import load_trained_model as load_svd_model, recommend_movies as recommend_svd
from recommender.ncf_model import load_trained_ncf_model, recommend_movies_ncf, load_data



# Load datasets
ratings = pd.read_csv("data/ratings.csv")
metadata = pd.read_csv("data/movies_metadata.csv", low_memory=False)
metadata = metadata[pd.to_numeric(metadata['id'], errors='coerce').notnull()]
metadata['id'] = metadata['id'].astype(int)

# Mapping for movie titles
id_to_title = dict(zip(metadata['id'], metadata['title']))

# Get common valid movie IDs
valid_movie_ids = set(metadata['id']).intersection(set(ratings['movieId']))
sample_movie_ids = list(pd.Series(list(valid_movie_ids)).sample(20, random_state=42))
user_ids = sorted(ratings['userId'].unique())

# Load models
svd_model = load_svd_model()
ncf_model = load_trained_ncf_model()
_, _, _, user_map, movie_map = load_data()

# UI
st.title("🎬 Personalized Movie Recommender")

model_choice = st.radio("Choose a model:", ["SVD (Matrix Factorization)", "NCF (Neural Collaborative Filtering)"])
selected_user = st.selectbox("Select a user:", user_ids)
top_n = st.slider("How many movies to recommend?", min_value=5, max_value=20, value=10)

if st.button("Get Recommendations"):
    st.subheader("Top Movie Recommendations:")

    if model_choice.startswith("SVD"):
        recommendations = recommend_svd(svd_model, selected_user, sample_movie_ids, n=top_n)
        for movie_id, score in recommendations:
            title = id_to_title.get(int(movie_id), f"Movie ID {movie_id}")
            st.markdown(f"🎥 **{title}** — Predicted Rating: `{score:.2f}`")

    else:
        recommendations = recommend_movies_ncf(ncf_model, selected_user, sample_movie_ids,
                                               user_map, movie_map, id_to_title, top_n=top_n)
        for title, score in recommendations:
            st.markdown(f"🤖 **{title}** — Predicted Rating: `{score:.2f}`")



from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split
from surprise import accuracy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import time


def evaluate_models_subset():
    st.subheader("📊 Model Performance Comparison")
    with st.spinner("Evaluating models..."):

        # Load subset for evaluation
        ratings_subset = ratings.sample(n=1000, random_state=42)  # Adjust size for speed

        true_ratings = []
        svd_preds = []
        ncf_preds = []

        for _, row in ratings_subset.iterrows():
            uid = int(row["userId"])
            mid = int(row["movieId"])
            true = row["rating"]

            # SVD prediction
            try:
                pred_svd = svd_model.predict(uid, mid).est
            except:
                continue

            # NCF prediction
            user_idx = user_map.get(uid)
            movie_idx = movie_map.get(mid)
            if user_idx is None or movie_idx is None:
                continue
            user_input = np.array([user_idx])
            movie_input = np.array([movie_idx])
            pred_ncf = ncf_model.predict([user_input, movie_input], verbose=0)[0][0]


            # Collect
            true_ratings.append(true)
            svd_preds.append(pred_svd)
            ncf_preds.append(pred_ncf)

        # Evaluation
        svd_rmse = np.sqrt(mean_squared_error(true_ratings, svd_preds))
        svd_mae = mean_absolute_error(true_ratings, svd_preds)

        ncf_rmse = np.sqrt(mean_squared_error(true_ratings, ncf_preds))
        ncf_mae = mean_absolute_error(true_ratings, ncf_preds)

        # Display metrics
        st.write("### 📈 RMSE & MAE")
        metrics_df = pd.DataFrame({
            "Model": ["SVD", "NCF"],
            "RMSE": [svd_rmse, ncf_rmse],
            "MAE": [svd_mae, ncf_mae]
        })
        st.dataframe(metrics_df)

        # Visualize
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width / 2, [svd_rmse, ncf_rmse], width, label="RMSE")
        ax.bar(x + width / 2, [svd_mae, ncf_mae], width, label="MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(["SVD", "NCF"])
        ax.set_title("Model Performance Comparison")
        ax.legend()
        st.pyplot(fig)


tab = st.sidebar.selectbox("Choose a tab", ["Recommendations", "Model Comparison"])

if tab == "Model Comparison":
    evaluate_models_subset()