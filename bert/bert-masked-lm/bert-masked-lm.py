import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103
) -> Tuple[np.ndarray, np.ndarray]:
    masked_ids = token_ids.copy()
    labels = np.full_like(token_ids, -100)

    labels[mask_positions] = token_ids[mask_positions]

    mask_mask = mask_positions & (replace_probs < 0.8)
    random_mask = mask_positions & (replace_probs >= 0.8) & (replace_probs < 0.9)

    masked_ids[mask_mask] = mask_token_id
    masked_ids[random_mask] = random_tokens[random_mask]

    return masked_ids, labels

class MLMHead:
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        return hidden_states @ self.W + self.b