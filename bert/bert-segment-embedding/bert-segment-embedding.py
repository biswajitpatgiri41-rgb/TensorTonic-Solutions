import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """

    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size

        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02

        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02

        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02

    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (batch_size, seq_len, hidden_size)
            containing the sum of token, position, and segment embeddings.
        """
        batch_size, seq_len = token_ids.shape

        # Token embeddings: (batch_size, seq_len, hidden_size)
        token_embeds = self.token_embeddings[token_ids]

        # Position embeddings: (seq_len, hidden_size)
        positions = np.arange(seq_len)
        position_embeds = self.position_embeddings[positions]

        # Broadcast to (batch_size, seq_len, hidden_size)
        position_embeds = np.broadcast_to(
            position_embeds, (batch_size, seq_len, self.hidden_size)
        )

        # Segment embeddings: (batch_size, seq_len, hidden_size)
        segment_embeds = self.segment_embeddings[segment_ids]

        # Combined BERT embeddings
        embeddings = token_embeds + position_embeds + segment_embeds

        return embeddings