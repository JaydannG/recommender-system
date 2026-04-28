import pandas as pd
import numpy as np

class PopularityRecommender:
    def __init__(self, min_ratings: int = 50, method: str = "simple"):
        self.min_ratings = min_ratings
        self.method = method
        self.movie_scores = None

    def _compute_score(self, df: pd.DataFrame):
        if self.method == "simple":
            return df["avg_rating"] * (
                df["num_ratings"] / df["num_ratings"].max()
            )
        elif self.method == "log":
            return df["avg_rating"] * np.log1p(df["num_ratings"])
        elif self.method == "bayesian":
            C = df["avg_rating"].mean()
            m = self.min_ratings
            v = df["num_ratings"]
            R = df["avg_rating"]

            return (v / (v + m)) * R + (m / (v + m)) * C
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, ratings: pd.DataFrame):
        movie_stats = (
            ratings.groupby("movie_id")
            .agg(
                avg_rating=("rating", "mean"),
                num_ratings=("rating", "count")
            )
            .reset_index()
        )

        movie_stats = movie_stats[movie_stats["num_ratings"] >= self.min_ratings]

        movie_stats["score"] = self._compute_score(movie_stats)

        self.movie_scores = movie_stats.sort_values(by="score", ascending=False)

        return self

    def recommend(self, user_id: int, ratings: pd.DataFrame, k: int = 10):
        if self.movie_scores is None:
            raise ValueError("Model has not been fitted yet.")

        seen_movies = set(ratings[ratings["user_id"] == user_id]["movie_id"])

        recommendations = self.movie_scores[
            ~self.movie_scores["movie_id"].isin(seen_movies)
        ]

        return recommendations.head(k)