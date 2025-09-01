# Movie Recommendation System

An advanced movie recommendation system that leverages Collaborative Filtering, Neural Collaborative Filtering (NCF), and Hybrid models to provide personalized movie suggestions to users.

## Project Overview

This system analyzes user-movie interactions to identify patterns and preferences, enabling accurate predictions of movies that users might enjoy. The project implements multiple recommendation algorithms and compares their performance to deliver the most effective recommendations.

## Features

- **Multiple Recommendation Algorithms**:
  - Collaborative Filtering (User-based and Item-based)
  - Matrix Factorization using Singular Value Decomposition (SVD)
  - Neural Collaborative Filtering (NCF)
  - Hybrid recommendation approaches

- **Comprehensive Evaluation**:
  - Performance metrics (RMSE, MAE, Precision, Recall)
  - Model comparison and analysis
  - Visualization of recommendation quality

- **User-Friendly Interface**:
  - Simple API for generating recommendations
  - Interactive demo application

## Project Structure


    movie-recommender-system/
    ├── .venv/ # Virtual environment (not included in repo)
    ├── app/ # Web application code
    ├── data/ # Dataset directory
    │ └── README.md # Instructions for obtaining datasets
    ├── models/ # Trained model files
    │ └── README.md # Instructions for generating models
    ├── notebooks/ # Jupyter notebooks for analysis and training
    ├── recommender/ # Core recommendation system code
    ├── utils/ # Utility functions
    ├── compare_models # Script to compare model performance
    ├── README.md # Project documentation
    ├── requirements.txt # Python dependencies
    └── run # Script to run the recommendation system


## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/guneet0329/movie-recommender-system.git
   cd movie-recommender-system

2. **Create and activate a virtual environment**:

    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # macOS/Linux
    python -m venv .venv
    source .venv/bin/activate

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt

4. **Download the datasets:**

    - The system uses the Kaggle dataset
    - Due to file size limitations, the full dataset is not included in the repository
    - Download from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
    - Place the ratings.csv file in the data/ directory


## Usage

- **Training Models**

    1. **Run the training notebooks:**

        - Navigate to the notebooks/ directory
        - Execute the model training notebooks in sequence
        - This will generate model files in the models/ directory

    2. **Compare model performance:**

        ```bash
        python compare_models

- **Generating Recommendations**

    1. **Using the command-line interface:**
        
        ```bash
        ./run --user_id 42 --num_recommendations 10

    2. **Using the web application:**
        
        ```bash
        cd app
        python app.py

    Then open your browser and navigate to http://localhost:5000
        

## Model Details

- **Collaborative Filtering**

    Analyzes user-item interactions to find similar users or items and makes recommendations based on these similarities.

- **Singular Value Decomposition (SVD)**

    A matrix factorization technique that decomposes the user-item interaction matrix into lower-dimensional matrices to uncover latent factors.

- **Neural Collaborative Filtering (NCF)**
    
    A deep learning approach that combines the linearity of matrix factorization with the non-linearity of neural networks for improved recommendation accuracy.

- **Hybrid Models**
    
    Combines multiple recommendation approaches to leverage the strengths of different algorithms.


## Performance Metrics
    
    The system evaluates recommendations using:

        - Root Mean Square Error (RMSE)
        - Mean Absolute Error (MAE)
        - Precision@K
        - Recall@K
        - F1 Score
    
## Future Improvements

    - Content-based filtering using movie metadata
    - Real-time recommendation updates
    - A/B testing framework for model evaluation
    - Deployment to cloud infrastructure