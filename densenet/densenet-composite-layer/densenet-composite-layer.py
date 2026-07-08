import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    x = torch.as_tensor(x, dtype=torch.float64)
    bn_gamma = torch.as_tensor(bn_gamma, dtype=torch.float64)
    bn_beta = torch.as_tensor(bn_beta, dtype=torch.float64)
    bn_mean = torch.as_tensor(bn_mean, dtype=torch.float64)
    bn_var = torch.as_tensor(bn_var, dtype=torch.float64)
    conv_weight = torch.as_tensor(conv_weight, dtype=torch.float64)

    y = (x - bn_mean.view(1, -1, 1, 1)) / torch.sqrt(bn_var.view(1, -1, 1, 1) + eps)
    y = y * bn_gamma.view(1, -1, 1, 1) + bn_beta.view(1, -1, 1, 1)
    y = F.relu(y)
    return F.conv2d(y, conv_weight, bias=None, stride=1, padding=1)