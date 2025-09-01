from recommender.ncf_model import load_data, build_ncf_model
from sklearn.model_selection import train_test_split
import numpy as np

# Load and preprocess data
df, num_users, num_movies, user_map, movie_map = load_data()
df = df.sample(10000, random_state=42).reset_index(drop=True)

# Split data
X = df[['user', 'movie']].values
y = df['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare input
train_users = X_train[:, 0]
train_movies = X_train[:, 1]
test_users = X_test[:, 0]
test_movies = X_test[:, 1]

# Build and train the model
model = build_ncf_model(num_users, num_movies)
model.fit([train_users, train_movies], y_train,
          validation_data=([test_users, test_movies], y_test),
          epochs=5,
          batch_size=128)

# Save the model (optional)
model.save("models/ncf_model.h5")
print("✅ NCF model trained and saved.")
