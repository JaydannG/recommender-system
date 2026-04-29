import pandas as pd

from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

def evaluate_model(model, train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 10):
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    relevant_items_by_user = test_df.groupby("user_id")["movie_id"].apply(list).to_dict()

    user_ids = list(relevant_items_by_user.keys())

    for user_id in user_ids:
        recs = model.recommend(user_id=user_id, ratings=train_df, k=k)
        recommended_items = recs["movie_id"].tolist()
        relevant_items = relevant_items_by_user[user_id]

        precision_scores.append(precision_at_k(recommended_items, relevant_items, k))
        recall_scores.append(recall_at_k(recommended_items, relevant_items, k))
        ndcg_scores.append(ndcg_at_k(recommended_items, relevant_items, k))

    return {
        "precision_at_k": sum(precision_scores) / len(precision_scores),
        "recall_at_k": sum(recall_scores) / len(recall_scores),
        "ndcg_at_k": sum(ndcg_scores) / len(ndcg_scores)
    }

def evaluate_model(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    sample_users: int | None = None,
    random_state: int = 42,
):
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    relevant_items_by_user = (
        test_df.groupby("user_id")["movie_id"]
        .apply(list)
        .to_dict()
    )

    user_ids = list(relevant_items_by_user.keys())

    if sample_users is not None:
        user_ids = (
            pd.Series(user_ids)
            .sample(n=sample_users, random_state=random_state)
            .tolist()
        )

    for user_id in user_ids:
        recs = model.recommend(user_id=user_id, ratings=train_df, k=k)
        recommended_items = recs["movie_id"].tolist()
        relevant_items = relevant_items_by_user[user_id]

        precision_scores.append(precision_at_k(recommended_items, relevant_items, k))
        recall_scores.append(recall_at_k(recommended_items, relevant_items, k))
        ndcg_scores.append(ndcg_at_k(recommended_items, relevant_items, k))

    return {
        "precision_at_k": sum(precision_scores) / len(precision_scores),
        "recall_at_k": sum(recall_scores) / len(recall_scores),
        "ndcg_at_k": sum(ndcg_scores) / len(ndcg_scores),
        "num_users_evaluated": len(user_ids),
    }