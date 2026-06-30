import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: (batch_size, seq_len, d_model)
        W_q, W_k, W_v: (d_model, d_model)
        W_o: (d_model, d_model)
        num_heads: number of attention heads

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    
    Q_proj = np.dot(Q, W_q)  
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)

    
    def split_heads(x):
        
        x = x.reshape(batch_size, seq_len, num_heads, d_k)
        
        return x.transpose(0, 2, 1, 3)

    Q_heads = split_heads(Q_proj)
    K_heads = split_heads(K_proj)
    V_heads = split_heads(V_proj)

    
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    attention_weights = softmax(scores, axis=-1)

    
    attention_output = np.matmul(attention_weights, V_heads)

    
    attention_output = attention_output.transpose(0, 2, 1, 3)
    
    concat_output = attention_output.reshape(batch_size, seq_len, d_model)

    
    output = np.dot(concat_output, W_o)

    return output