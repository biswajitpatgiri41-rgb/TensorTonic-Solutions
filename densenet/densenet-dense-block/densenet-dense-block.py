import torch
import torch.nn.functional as F

def dense_block(x, layers, growth_rate, eps=1e-5):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    feats = [x]

    for layer in layers:
        z = torch.cat(feats, dim=1)

        gamma = torch.as_tensor(layer["bn_gamma"], dtype=z.dtype, device=z.device).view(1, -1, 1, 1)
        beta = torch.as_tensor(layer["bn_beta"], dtype=z.dtype, device=z.device).view(1, -1, 1, 1)
        mean = torch.as_tensor(layer["bn_mean"], dtype=z.dtype, device=z.device).view(1, -1, 1, 1)
        var = torch.as_tensor(layer["bn_var"], dtype=z.dtype, device=z.device).view(1, -1, 1, 1)
        weight = torch.as_tensor(layer["conv_weight"], dtype=z.dtype, device=z.device)

        z = gamma * (z - mean) / torch.sqrt(var + eps) + beta
        z = F.relu(z)
        z = F.conv2d(z, weight, bias=None, stride=1, padding=1)

        feats.append(z)

    return torch.cat(feats, dim=1)