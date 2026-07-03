import numpy as np
from typing import List, Tuple

def create_nsp_pairs(
    documents: List[List[str]],
    pair_specs: List[dict]
) -> List[Tuple[str, str, int]]:
    pairs = []
    for spec in pair_specs:
        doc_a = spec["doc_a"]
        doc_b = spec["doc_b"]
        sent_a = spec["sent_a"]
        sent_b = spec["sent_b"]

        sentence_a = documents[doc_a][sent_a]
        sentence_b = documents[doc_b][sent_b]
        label = int(doc_a == doc_b and sent_b == sent_a + 1)

        pairs.append((sentence_a, sentence_b, label))
    return pairs

class NSPHead:
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)

    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        return cls_hidden @ self.W + self.b

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)