import torch
import torch.nn.functional as F

def bottleneck_layer(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight,
                     bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight, eps=1e-5):
    x = torch.as_tensor(x, dtype=torch.float64)
    bn1_gamma = torch.as_tensor(bn1_gamma, dtype=torch.float64)
    bn1_beta = torch.as_tensor(bn1_beta, dtype=torch.float64)
    bn1_mean = torch.as_tensor(bn1_mean, dtype=torch.float64)
    bn1_var = torch.as_tensor(bn1_var, dtype=torch.float64)
    conv1_weight = torch.as_tensor(conv1_weight, dtype=torch.float64)
    bn2_gamma = torch.as_tensor(bn2_gamma, dtype=torch.float64)
    bn2_beta = torch.as_tensor(bn2_beta, dtype=torch.float64)
    bn2_mean = torch.as_tensor(bn2_mean, dtype=torch.float64)
    bn2_var = torch.as_tensor(bn2_var, dtype=torch.float64)
    conv2_weight = torch.as_tensor(conv2_weight, dtype=torch.float64)

    y = (x - bn1_mean.view(1, -1, 1, 1)) / torch.sqrt(bn1_var.view(1, -1, 1, 1) + eps)
    y = y * bn1_gamma.view(1, -1, 1, 1) + bn1_beta.view(1, -1, 1, 1)
    y = F.relu(y)
    y = F.conv2d(y, conv1_weight, bias=None, stride=1, padding=0)

    y = (y - bn2_mean.view(1, -1, 1, 1)) / torch.sqrt(bn2_var.view(1, -1, 1, 1) + eps)
    y = y * bn2_gamma.view(1, -1, 1, 1) + bn2_beta.view(1, -1, 1, 1)
    y = F.relu(y)
    y = F.conv2d(y, conv2_weight, bias=None, stride=1, padding=1)

    return y