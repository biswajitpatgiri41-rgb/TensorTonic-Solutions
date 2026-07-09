import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    classes = np.unique(y_train)
    n_train = X_train.shape[0]
    log_posteriors = np.zeros((X_test.shape[0], len(classes)))

    for idx, c in enumerate(classes):
        X_c = X_train[y_train == c]
        n_c = X_c.shape[0]

        log_prior = np.log(n_c / n_train)

        theta = (X_c.sum(axis=0) + 1) / (n_c + 2)

        log_theta = np.log(theta)
        log_one_minus_theta = np.log(1 - theta)

        log_posteriors[:, idx] = (
            log_prior
            + X_test @ log_theta
            + (1 - X_test) @ log_one_minus_theta
        )

    return log_posteriors