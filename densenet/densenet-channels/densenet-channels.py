import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int, block_layers, compression: float) -> torch.Tensor:
    channels = stem_channels
    counts = [channels]

    for i, n in enumerate(block_layers):
        channels += n * growth_rate
        counts.append(channels)
        if i != len(block_layers) - 1:
            channels = int(math.floor(channels * compression))
            counts.append(channels)

    return torch.tensor(counts, dtype=torch.int64)