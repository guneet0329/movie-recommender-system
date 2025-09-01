from recommender.collaborative import train_svd_model, load_trained_model, recommend_movies
import pandas as pd

# Load movie titles
movies_metadata_df = pd.read_csv("data/movies_metadata.csv", low_memory=False)
id_to_title = dict(zip(movies_metadata_df['id'], movies_metadata_df['title']))

# Load model instead of retraining
#algo = train_svd_model("data/ratings.csv")
algo = load_trained_model()

# Sample movie IDs from metadata
movie_ids = [862, 8844, 15602, 31357, 11862]
recommendations = recommend_movies(algo, user_id=1, movie_ids=movie_ids)

print("Top Recommendations:")
for movie_id, score in recommendations:
    title = id_to_title.get(int(movie_id), f"Movie ID {movie_id}")
    print(f"{title} – Predicted Rating: {score:.2f}")
