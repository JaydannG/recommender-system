import pandas as pd

class PopularityRecommender:
    def __init__(self, min_ratings: int = 50):
        self.min_rating = min_ratings
        self.movie_scores = None

    def fit(self, ratings: pd.DataFrame):
        movie_stats = (
            ratings.groupby("movie_id")
            .agg(
                avg_rating=("rating", "mean"),
                num_ratings=("rating", "count")
            )
            .reset_index()
        )

        movie_stats = movie_stats[movie_stats["num_ratings"] >= self.min_rating]

        movie_stats["score"] = movie_stats["avg_rating"] * (
            movie_stats["num_ratings"] / movie_stats["num_ratings"].max()
        )

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