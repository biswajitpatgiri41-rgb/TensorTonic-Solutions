def remove_stopwords(tokens, stopwords):
    stopword_set = set(stopwords)
    return [token for token in tokens if token not in stopword_set]