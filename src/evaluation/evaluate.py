import pandas as pd

from src.evaluation.metrics import precision_at_k

def evaluate_precision_at_k(model, train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 10):
    scores = []

    for user_id in test_df["user_id"].unique():
        recs = model.recommend(user_id=user_id, ratings=train_df, k=k)
        recommended_items = recs["movie_id"].tolist()
        relevant_items = test_df[test_df["user_id"] == user_id]["movie_id"].tolist()

        score = precision_at_k(recommended_items, relevant_items, k)
        scores.append(score)

    return sum(scores) / len(scores)