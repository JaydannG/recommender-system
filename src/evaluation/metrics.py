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

def ndcg_at_k(recommended_items, relevant_items, k: int = 10):
    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)

    dcg = 0.0

    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            rank = i + 1
            dcg += 1 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)

    if ideal_hits == 0:
        return 0.0
    
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg