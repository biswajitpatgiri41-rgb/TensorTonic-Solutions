import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    
    if not documents:
        return np.array([]), []

    N = len(documents)
    
    
    tokenized_docs = []
    vocab_set = set()
    
    for doc in documents:
        tokens = doc.lower().split()
        tokenized_docs.append(tokens)
        vocab_set.update(tokens)
        
    
    if not vocab_set:
        return np.zeros((N, 0)), []
        
    
    vocabulary = sorted(list(vocab_set))
    
    
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    n_vocab = len(vocabulary)
    
    
    df_counts = Counter()
    for tokens in tokenized_docs:
        
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df_counts[token] += 1
            
    
    idf_array = np.zeros(n_vocab)
    for word, idx in word_to_index.items():
        idf_array[idx] = math.log(N / df_counts[word])
        
    
    tfidf_matrix = np.zeros((N, n_vocab))
    
    for i, tokens in enumerate(tokenized_docs):
        total_terms = len(tokens)
        
        
        if total_terms == 0:
            continue
            
        
        term_counts = Counter(tokens)
        for word, count in term_counts.items():
            tf = count / total_terms
            idx = word_to_index[word]
            
            
            tfidf_matrix[i, idx] = tf * idf_array[idx]
            
    return tfidf_matrix, vocabulary