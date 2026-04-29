import pandas as pd

class HybridRecommender:
    def __init__(
        self,
        popularity_model,
        content_model,
        collaborative_model,
        weights=None
    ):
        self.popularity_model = popularity_model
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.weights = weights or {
            "popularity": 0.15,
            "content": 0.25,
            "collaborative": 0.60
        }

    def fit(self, train_df: pd.DataFrame, movies: pd.DataFrame, genre_matrix: pd.DataFrame):
        self.popularity_model.fit(train_df)
        self.content_model.fit(movies, genre_matrix)
        self.collaborative_model.fit(train_df)

        return self

    def _normalize_scores(self, recs: pd.DataFrame, score_name: str):
        if recs.empty:
            return pd.DataFrame(columns=["movie_id", score_name])

        recs = recs[["movie_id", "score"]].copy()

        min_score = recs["score"].min()
        max_score = recs["score"].max()

        if max_score == min_score:
            recs[score_name] = 1.0
        else:
            recs[score_name] = (recs["score"] - min_score) / (max_score - min_score)

        return recs[["movie_id", score_name]]

    def recommend(self, user_id, ratings, k = 10):
        candidate_k = 100

        pop_recs = self.popularity_model.recommend(user_id, ratings, k=candidate_k)
        content_recs = self.content_model.recommend(user_id, ratings, k=candidate_k)
        collab_recs = self.collaborative_model.recommend(user_id, ratings, k=candidate_k)

        pop_recs = self._normalize_scores(pop_recs, "popularity_score")
        content_recs = self._normalize_scores(content_recs, "content_score")
        collab_recs = self._normalize_scores(collab_recs, "collaborative_score")

        merged = (
            pop_recs
            .merge(content_recs, on="movie_id", how="outer")
            .merge(collab_recs, on="movie_id", how="outer")
            .fillna(0)
        )

        if merged.empty:
            return pd.DataFrame(columns=["movie_id", "final_score"])

        merged["score"] = (
            self.weights["popularity"] * merged["popularity_score"] +
            self.weights["content"] * merged["content_score"] +
            self.weights["collaborative"] * merged["collaborative_score"]
        )

        return merged.sort_values("score", ascending=False).head(k)[["movie_id", "score"]]