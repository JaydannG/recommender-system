import pandas as pd

from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

def evaluate_precision_at_k(model, train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 10):
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for user_id in test_df["user_id"].unique():
        recs = model.recommend(user_id=user_id, ratings=train_df, k=k)
        recommended_items = recs["movie_id"].tolist()
        relevant_items = test_df[test_df["user_id"] == user_id]["movie_id"].tolist()

        precision_scores.append(precision_at_k(recommended_items, relevant_items, k))
        recall_scores.append(recall_at_k(recommended_items, relevant_items, k))
        ndcg_scores.append(ndcg_at_k(recommended_items, relevant_items, k))

    return {
        "precision_at_k": sum(precision_scores) / len(precision_scores),
        "recall_at_k": sum(recall_scores) / len(recall_scores),
        "ndcg_at_k": sum(ndcg_scores) / len(ndcg_scores)
    }