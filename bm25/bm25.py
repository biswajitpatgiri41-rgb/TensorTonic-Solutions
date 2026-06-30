import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    
    if not docs:
        return np.array([], dtype=float)

    N = len(docs)
    
    
    doc_lengths = np.array([len(doc) for doc in docs])
    avgdl = np.mean(doc_lengths)
    
    
    if avgdl == 0:
        avgdl = 1.0 
    
    
    unique_query_terms = list(dict.fromkeys(query_tokens))
    
    
    df_counts = Counter()
    for doc in docs:
        
        unique_terms = set(doc)
        for term in unique_query_terms:
            if term in unique_terms:
                df_counts[term] += 1
                
    
    idfs = {}
    for term in unique_query_terms:
        df_t = df_counts.get(term, 0)
        
        if df_t == 0:
            idfs[term] = 0.0
        else:
            
            idfs[term] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0)
            
    
    scores = np.zeros(N, dtype=float)
    
    
    len_norm = 1.0 - b + b * (doc_lengths / avgdl)
    
    for i, doc in enumerate(docs):
        
        term_freqs = Counter(doc)
        doc_score = 0.0
        
        for term in unique_query_terms:
            tf = term_freqs.get(term, 0)
            
            
            if tf > 0:
                
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * len_norm[i]
                
                
                doc_score += idfs[term] * (numerator / denominator)
                
        scores[i] = doc_score
        
    return scores