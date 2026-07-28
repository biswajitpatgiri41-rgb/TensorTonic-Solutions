import math

def perplexity(prob_distributions, actual_tokens):
    log_sum = 0.0
    n = len(actual_tokens)
    for i in range(n):
        log_sum += math.log(prob_distributions[i][actual_tokens[i]])
    cross_entropy = -log_sum / n
    return math.exp(cross_entropy)