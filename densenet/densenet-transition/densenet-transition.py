import torch
import torch.nn.functional as F

def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    x = torch.as_tensor(x, dtype=torch.float32)
    bn_gamma = torch.as_tensor(bn_gamma, dtype=x.dtype, device=x.device)
    bn_beta = torch.as_tensor(bn_beta, dtype=x.dtype, device=x.device)
    bn_mean = torch.as_tensor(bn_mean, dtype=x.dtype, device=x.device)
    bn_var = torch.as_tensor(bn_var, dtype=x.dtype, device=x.device)
    conv_weight = torch.as_tensor(conv_weight, dtype=x.dtype, device=x.device)

    bn_gamma = bn_gamma.reshape(1, -1, 1, 1)
    bn_beta = bn_beta.reshape(1, -1, 1, 1)
    bn_mean = bn_mean.reshape(1, -1, 1, 1)
    bn_var = bn_var.reshape(1, -1, 1, 1)

    x = (x - bn_mean) / torch.sqrt(bn_var + eps)
    x = bn_gamma * x + bn_beta
    x = F.relu(x)
    x = F.conv2d(x, conv_weight, bias=None, stride=1, padding=0)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)

    return x