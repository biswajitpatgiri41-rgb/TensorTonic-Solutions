def item_cf_predict(user_ratings, item_similarities, target):
    numerator = 0.0
    denominator = 0.0

    for i in range(len(user_ratings)):
        if i == target:
            continue
        if user_ratings[i] != 0 and item_similarities[i] > 0:
            numerator += item_similarities[i] * user_ratings[i]
            denominator += item_similarities[i]

    if denominator == 0:
        return 0.0

    return numerator / denominator