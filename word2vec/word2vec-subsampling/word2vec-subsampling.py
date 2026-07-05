import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    counts = counts.float()
    f = counts / counts.sum()
    return torch.clamp(torch.sqrt(torch.tensor(t, dtype=f.dtype, device=f.device) / f), max=1.0)