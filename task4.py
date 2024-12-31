import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import streamlit as st

# Load the dataset (you can replace this with your file path)
ratings = pd.read_csv('ratings.csv')  # Replace with your dataset file
movies = pd.read_csv('movies.csv')    # Replace with your dataset file

# Create a user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Fill missing values with 0 (or another strategy depending on the model)
user_item_matrix = user_item_matrix.fillna(0)

# Split data into train and test sets
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

# Apply NMF for matrix factorization
n_factors = 20  # Number of latent features
nmf_model = NMF(n_components=n_factors, init='random', random_state=42)

# Train the model
user_features = nmf_model.fit_transform(train_data)
item_features = nmf_model.components_

# Reconstruct the matrix
predicted_ratings = np.dot(user_features, item_features)

# Convert to DataFrame for easier interpretation
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=train_data.columns, index=train_data.index)

# Evaluate the model using Mean Squared Error (MSE)
test_ratings = test_data.values
mse = mean_squared_error(test_ratings[test_ratings > 0], predicted_ratings[test_ratings > 0])
print(f'Mean Squared Error: {mse}')

# Function to recommend top N movies for a user
def recommend_movies(user_id, n=10):
    user_row = predicted_ratings_df.loc[user_id]
    top_n_movies = user_row.sort_values(ascending=False).head(n)
    recommended_movie_ids = top_n_movies.index
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    return recommended_movies[['movieId', 'title']]

# Create Streamlit app interface
st.title('Movie Recommendation System')

# Input user ID for recommendations
user_id = st.number_input('Enter User ID', min_value=1, max_value=610, step=1)

# Recommend movies based on user input
recommended_movies = recommend_movies(user_id=user_id, n=10)

# Display recommended movies
st.write(f"Recommended Movies for User {user_id}")
st.write(recommended_movies)

