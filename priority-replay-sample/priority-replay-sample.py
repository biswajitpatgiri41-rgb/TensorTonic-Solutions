def priority_replay_sample(priorities, alpha, beta):
    powered = [p ** alpha for p in priorities]
    total = sum(powered)
    probabilities = [p / total for p in powered]
    n = len(priorities)
    raw_weights = [(n * prob) ** (-beta) for prob in probabilities]
    max_weight = max(raw_weights)
    normalized_weights = [w / max_weight for w in raw_weights]
    return [probabilities, normalized_weights]