import pandas as pd

class ColdStartHybridRecommender:
    def __init__(
        self,
        hybrid_model,
        content_model,
        popularity_model,
        min_user_ratings: int = 10
    ):
        self.hybrid_model = hybrid_model
        self.content_model = content_model
        self.popularity_model = popularity_model
        self.min_user_ratings = min_user_ratings

    def fit(self, train_df: pd.DataFrame, movies: pd.DataFrame, genre_matrix: pd.DataFrame):
        self.hybrid_model.fit(train_df, movies, genre_matrix)
        self.content_model.fit(movies, genre_matrix)
        self.popularity_model.fit(train_df)

        return self

    def recommend(self, user_id, ratings, k=10):
        user_ratings = ratings[ratings["user_id"] == user_id]
        num_ratings = len(user_ratings)

        if num_ratings == 0:
            return self.popularity_model.recommend(user_id, ratings, k=k)

        if num_ratings < self.min_user_ratings:
            content_recs = self.content_model.recommend(user_id, ratings, k=k)

            if not content_recs.empty:
                return content_recs
            
            return self.popularity_model.recommend(user_id, ratings, k=k)

        return self.hybrid_model.recommend(user_id, ratings, k=k)