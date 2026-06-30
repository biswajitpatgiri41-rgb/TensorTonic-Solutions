import torch
import torch.nn as nn
import math


def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer with vocab_size embeddings of dimension d_model.
    """
    
    embedding = nn.Embedding(vocab_size, d_model)
    return embedding


def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.

    Args:
        embedding: nn.Embedding layer
        tokens: 1D tensor of token indices (sequence_length,)
        d_model: embedding dimension

    Returns:
        Tensor of shape (sequence_length, d_model)
    """
    
    embedded = embedding(tokens)  

    
    scaled_embedding = embedded * math.sqrt(d_model)

    return scaled_embedding