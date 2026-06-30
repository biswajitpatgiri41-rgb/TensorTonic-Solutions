import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape (batch, seq_len_q, d_k)
        K: Key tensor of shape (batch, seq_len_k, d_k)
        V: Value tensor of shape (batch, seq_len_k, d_v)

    Returns:
        Tensor of shape (batch, seq_len_q, d_v)
    """

    
    scores = torch.matmul(Q, K.transpose(-2, -1))

    
    d_k = K.shape[-1]
    scores = scores / math.sqrt(d_k)

    
    attention_weights = F.softmax(scores, dim=-1)

    
    output = torch.matmul(attention_weights, V)

    return output