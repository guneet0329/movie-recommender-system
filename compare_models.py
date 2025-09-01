from recommender.collaborative import load_trained_model
from recommender.ncf_model import load_trained_ncf_model, load_data
import pandas as pd
import numpy as np
from surprise import accuracy
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split


def evaluate_svd_model(ratings_csv_path="data/ratings.csv"):
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    data = Dataset.load_from_file(ratings_csv_path, reader=reader)
    trainset, testset = surprise_split(data, test_size=0.2, random_state=42)

    svd = load_trained_model()
    predictions = svd.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    return rmse, mae


from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_ncf_model():
    model = load_trained_ncf_model()
    X_train, X_test, y_train, y_test, *_ = load_data()

    y_pred = model.predict(X_test).flatten()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    return rmse, mae


svd_rmse, svd_mae = evaluate_svd_model()
ncf_rmse, ncf_mae = evaluate_ncf_model()

print(f"SVD → RMSE: {svd_rmse:.4f}, MAE: {svd_mae:.4f}")
print(f"NCF → RMSE: {ncf_rmse:.4f}, MAE: {ncf_mae:.4f}")


#Graph

import matplotlib.pyplot as plt

labels = ['SVD', 'NCF']
rmse_vals = [svd_rmse, ncf_rmse]
mae_vals = [svd_mae, ncf_mae]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, rmse_vals, width, label='RMSE')
ax.bar(x + width/2, mae_vals, width, label='MAE')

ax.set_ylabel('Error')
ax.set_title('Model Comparison (Lower is Better)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.grid(True)
plt.tight_layout()
plt.show()
