def user_based_cf_prediction(similarities, ratings):
    num = 0.0
    den = 0.0
    for sim, rating in zip(similarities, ratings):
        if sim > 0:
            num += sim * rating
            den += sim
    return 0.0 if den == 0 else num / den