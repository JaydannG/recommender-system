import pandas as pd
import numpy as np

class ContentRecommender:
    def __init__(self):
        self.genre_matrix = None
        self.movie_features = None

    def fit(self, movies: pd.DataFrame, genre_matrix: pd.DataFrame):
        self.movie_features = genre_matrix.set_index("movie_id")
        return self

    def _build_user_profile(self, user_id, ratings, min_rating=4):
        user_ratings = ratings[
            (ratings["user_id"] == user_id) & (ratings["rating"] >= min_rating)
        ]

        if user_ratings.empty:
            return None
        
        liked_movies = user_ratings["movie_id"]

        user_profile = self.movie_features.loc[liked_movies].mean(axis=0)

        return user_profile

    def recommend(self, user_id, ratings, k=10):
        user_profile = self._build_user_profile(user_id, ratings)

        if user_profile is None:
            return pd.DataFrame(columns=["movie_id", "score"])

        scores = self.movie_features.dot(user_profile)

        if scores.empty:
            return pd.DataFrame(columns=["movie_id", "score"])

        seen_movies = set(ratings[ratings["user_id"] == user_id]["movie_id"])
        scores = scores[~scores.index.isin(seen_movies)]
        
        recs = scores.sort_values(ascending=False).head(k).reset_index()
        recs.columns = ["movie_id", "score"]

        return recs