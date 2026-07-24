import numpy as np

def cohens_kappa(rater1, rater2):
    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)
    n = len(rater1)

    p_o = np.mean(rater1 == rater2)

    labels = np.union1d(rater1, rater2)
    p_e = 0.0
    for label in labels:
        p1 = np.sum(rater1 == label) / n
        p2 = np.sum(rater2 == label) / n
        p_e += p1 * p2

    if p_e == 1.0:
        return 1.0

    return float((p_o - p_e) / (1.0 - p_e))