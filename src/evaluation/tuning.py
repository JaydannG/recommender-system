import pandas as pd

from src.evaluation.evaluate import evaluate_model
from src.models.hybrid import HybridRecommender
from src.models.popularity import PopularityRecommender
from src.models.content import ContentRecommender
from src.models.item_collaborative import ItemCollaborativeRecommender


def tune_hybrid_weights(
    train_df,
    test_df,
    movies,
    genre_matrix,
    k: int = 10,
    sample_users: int | None = 500,
):
    weight_configs = {
        "collab_90": {
            "popularity": 0.05,
            "content": 0.05,
            "collaborative": 0.90,
        },
        "collab_85": {
            "popularity": 0.05,
            "content": 0.10,
            "collaborative": 0.85,
        },
        "collab_80": {
            "popularity": 0.10,
            "content": 0.10,
            "collaborative": 0.80,
        },
        "collab_75": {
            "popularity": 0.10,
            "content": 0.15,
            "collaborative": 0.75,
        },
        "collab_70": {
            "popularity": 0.15,
            "content": 0.15,
            "collaborative": 0.70,
        },
        "balanced": {
            "popularity": 0.33,
            "content": 0.33,
            "collaborative": 0.34,
        },
    }

    results = []

    for name, weights in weight_configs.items():
        model = HybridRecommender(
            popularity_model=PopularityRecommender(method="bayesian"),
            content_model=ContentRecommender(),
            collaborative_model=ItemCollaborativeRecommender(min_rating=4),
            weights=weights,
        )

        model.fit(train_df, movies, genre_matrix)

        metrics = evaluate_model(
            model,
            train_df,
            test_df,
            k=k,
            sample_users=sample_users,
        )

        metrics["model"] = name
        metrics["popularity_weight"] = weights["popularity"]
        metrics["content_weight"] = weights["content"]
        metrics["collaborative_weight"] = weights["collaborative"]

        results.append(metrics)

    results_df = pd.DataFrame(results)

    return results_df.sort_values("ndcg_at_k", ascending=False)