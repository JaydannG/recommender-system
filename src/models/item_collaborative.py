import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class ItemCollaborativeRecommender:
    def __init__(self, min_rating=4):
        self.min_rating = min_rating
        self.user_to_index = None
        self.movie_to_index = None
        self.index_to_movie = None
        self.item_similarity = None
        self.movie_ids = None

    def fit(self, ratings: pd.DataFrame):
        self.movie_ids = sorted(ratings["movie_id"].unique())
        user_ids = sorted(ratings["user_id"].unique())

        self.user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.movie_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        self.index_to_movie = {idx: movie_id for movie_id, idx in self.movie_to_index.items()}

        rows = ratings["user_id"].map(self.user_to_index)
        cols = ratings["movie_id"].map(self.movie_to_index)
        values = ratings["rating"]

        user_item_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(user_ids), len(self.movie_ids))
        )

        item_user_matrix = user_item_matrix.T

        self.item_similarity = cosine_similarity(item_user_matrix)

        return self

    def recommend(self, user_id, ratings, k=10):
        if self.item_similarity is None:
            raise ValueError("Model has not been fitted yet.")

        user_ratings = ratings[ratings["user_id"] == user_id]

        if user_ratings.empty:
            return pd.DataFrame(columns=["movie_id", "score"])

        liked_movies = user_ratings[user_ratings["rating"] >= self.min_rating]

        if liked_movies.empty:
            return pd.DataFrame(columns=["movie_id", "score"])

        seen_movies = set(user_ratings["movie_id"])

        scores = np.zeros(len(self.movie_ids))

        for _, row in liked_movies.iterrows():
            movie_id = row["movie_id"]
            rating = row["rating"]

            if movie_id not in self.movie_to_index:
                continue

            movie_idx = self.movie_to_index[movie_id]
            scores += self.item_similarity[movie_idx] * rating

        recs = []

        for idx, score in enumerate(scores):
            movie_id = self.index_to_movie[idx]

            if movie_id not in seen_movies:
                recs.append((movie_id, score))

        recs_df = pd.DataFrame(recs, columns=["movie_id", "score"])

        if recs_df.empty:
            return pd.DataFrame(columns=["movie_id", "score"])

        return recs_df.sort_values("score", ascending=False).head(k)