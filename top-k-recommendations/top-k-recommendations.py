def top_k_recommendations(scores, rated_indices, k):
    items = [(scores[i], i) for i in range(len(scores)) if i not in rated_indices]
    items.sort(key=lambda x: x[0], reverse=True)
    return [i for _, i in items[:k]]