import numpy as np

def bag_of_words_vector(tokens, vocab):
    word_to_index = {word: i for i, word in enumerate(vocab)}
    vector = np.zeros(len(vocab), dtype=int)
    for token in tokens:
        if token in word_to_index:
            vector[word_to_index[token]] += 1
    return vector