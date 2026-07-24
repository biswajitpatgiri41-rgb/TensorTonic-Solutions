def word_count_dict(sentences):
    word_counts = {}
    for sentence in sentences:
        for word in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts