import math

def precision_at_k(recommended_items, relevant_items, k: int = 10):
    recommended_k = recommended_items[:k]

    if (len(recommended_k) == 0):
        return 0.0

    recommended_set = set(recommended_k)
    relevant_set = set(relevant_items)

    hits = recommended_set.intersection(relevant_set)

    return len(hits) / k

def recall_at_k(recommended_items, relevant_items, k: int = 10):
    recommended_k = recommended_items[:k]

    if (len(recommended_k) == 0):
        return 0.0

    recommended_set = set(recommended_k)
    relevant_set = set(relevant_items)

    hits = recommended_set.intersection(relevant_set)

    return len(hits) / len(relevant_set)