import math

def ndcg(relevance_scores, k):
    k = min(k, len(relevance_scores))

    def dcg(scores):
        total = 0.0
        for i in range(k):
            total += (2 ** scores[i] - 1) / math.log2(i + 2)
        return total

    dcg_score = dcg(relevance_scores)
    idcg_score = dcg(sorted(relevance_scores, reverse=True))

    if idcg_score == 0:
        return 0.0

    return dcg_score / idcg_score