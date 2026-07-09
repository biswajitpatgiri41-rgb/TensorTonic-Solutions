def bigram_probabilities(tokens):
    counts = {}
    vocab = sorted(set(tokens))
    V = len(vocab)

    context_counts = {}

    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        counts[(w1, w2)] = counts.get((w1, w2), 0) + 1
        context_counts[w1] = context_counts.get(w1, 0) + 1

    probs = {}

    for w1 in vocab:
        denom = context_counts.get(w1, 0) + V
        for w2 in vocab:
            c = counts.get((w1, w2), 0)
            probs[(w1, w2)] = (c + 1) / denom

    return counts, probs