import math

def gaussian_naive_bayes(X_train, y_train, X_test):
    eps = 1e-9
    n = len(X_train)
    classes = sorted(set(y_train))
    stats = {}

    for c in classes:
        Xc = [X_train[i] for i in range(n) if y_train[i] == c]
        nc = len(Xc)
        d = len(Xc[0])

        means = []
        variances = []

        for j in range(d):
            mean = sum(row[j] for row in Xc) / nc
            var = sum((row[j] - mean) ** 2 for row in Xc) / nc + eps
            means.append(mean)
            variances.append(var)

        stats[c] = (nc / n, means, variances)

    predictions = []

    for x in X_test:
        best_class = None
        best_log_prob = -float("inf")

        for c in classes:
            prior, means, variances = stats[c]
            log_prob = math.log(prior)

            for j in range(len(x)):
                var = variances[j]
                mean = means[j]
                log_prob += -0.5 * math.log(2 * math.pi * var) - ((x[j] - mean) ** 2) / (2 * var)

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = c

        predictions.append(best_class)

    return predictions