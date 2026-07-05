import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    pairs = []
    n = token_ids.numel()
    for i in range(n):
        left = max(0, i - window)
        right = min(n - 1, i + window)
        for j in range(left, right + 1):
            if j != i:
                pairs.append([token_ids[i].item(), token_ids[j].item()])
    if not pairs:
        return torch.zeros((0, 2), dtype=torch.int64)
    return torch.tensor(pairs, dtype=torch.int64)