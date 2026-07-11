from collections import Counter
import math

def bleu_score(candidate, reference, max_n):
    if not candidate:
        return 0.0

    c_len = len(candidate)
    r_len = len(reference)
    precisions = []

    for n in range(1, max_n + 1):
        if c_len < n:
            return 0.0

        cand_ngrams = Counter(tuple(candidate[i:i + n]) for i in range(c_len - n + 1))
        ref_ngrams = Counter(tuple(reference[i:i + n]) for i in range(r_len - n + 1))

        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        p = clipped / total

        if p == 0:
            return 0.0

        precisions.append(p)

    bp = 1.0 if c_len >= r_len else math.exp(1 - r_len / c_len)
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)