import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if not seqs:
        return np.empty((0, 0), dtype=int)
    
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)
        
    result = np.full((len(seqs), max_len), pad_value, dtype=int)
    
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        if length > 0:
            result[i, :length] = seq[:length]
            
    return result