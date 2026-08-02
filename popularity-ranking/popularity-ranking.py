def popularity_ranking(items, min_votes, global_mean):
    return [
        (votes / (votes + min_votes)) * rating +
        (min_votes / (votes + min_votes)) * global_mean
        for rating, votes in items
    ]