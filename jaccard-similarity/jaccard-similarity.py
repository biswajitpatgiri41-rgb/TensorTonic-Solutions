def jaccard_similarity(set_a, set_b):
    a = set(set_a)
    b = set(set_b)
    union = a | b
    if len(union) == 0:
        return 0.0
    return len(a & b) / len(union)