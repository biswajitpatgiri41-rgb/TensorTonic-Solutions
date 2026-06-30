import numpy as np

def softmax(x, axis=-1):
    
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    
    
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    
    Q_proj = np.dot(Q, W_q)
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)

    
    def split_heads(x):
        x = x.reshape(batch_size, seq_len, num_heads, d_k)
        return np.transpose(x, (0, 2, 1, 3))

    Q_heads = split_heads(Q_proj)
    K_heads = split_heads(K_proj)
    V_heads = split_heads(V_proj)

    
    scores = np.matmul(Q_heads, np.transpose(K_heads, (0, 1, 3, 2)))
    scores = scores / np.sqrt(d_k)

    attn_weights = softmax(scores, axis=-1)
    attn_output = np.matmul(attn_weights, V_heads)

    
    attn_output = np.transpose(attn_output, (0, 2, 1, 3))
    attn_output = attn_output.reshape(batch_size, seq_len, d_model)

    
    output = np.dot(attn_output, W_o)

    return output

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    
    hidden = np.dot(x, W1) + b1
    hidden = np.maximum(0, hidden)  
    output = np.dot(hidden, W2) + b2
    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    
   
    attn_output = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + attn_output, gamma1, beta1)

    
    ffn_output = feed_forward(x, W1, b1, W2, b2)
    output = layer_norm(x + ffn_output, gamma2, beta2)

    return output