import torch
import torch.nn.functional as F


def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    x = F.batch_norm(
        x,
        running_mean=bn_mean,
        running_var=bn_var,
        weight=bn_gamma,
        bias=bn_beta,
        training=False,
        eps=eps,
    )
    x = F.relu(x)
    x = F.conv2d(x, conv_weight, bias=None, stride=1, padding=1)
    return x


def dense_block(x, layers, eps):
    feats = [x]
    for layer in layers:
        inp = torch.cat(feats, dim=1)
        out = composite_layer(
            inp,
            layer["bn_gamma"],
            layer["bn_beta"],
            layer["bn_mean"],
            layer["bn_var"],
            layer["conv_weight"],
            eps,
        )
        feats.append(out)
    return torch.cat(feats, dim=1)


def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    x = F.batch_norm(
        x,
        running_mean=bn_mean,
        running_var=bn_var,
        weight=bn_gamma,
        bias=bn_beta,
        training=False,
        eps=eps,
    )
    x = F.relu(x)
    x = F.conv2d(x, conv_weight, bias=None, stride=1, padding=0)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    return x


def densenet_forward(x, weights, growth_rate, eps=1e-5):
    x = F.conv2d(x, weights["stem_conv"], bias=None, stride=1, padding=1)

    blocks = weights["blocks"]
    transitions = weights["transitions"]

    for i, block in enumerate(blocks):
        x = dense_block(x, block, eps)
        if i < len(blocks) - 1:
            t = transitions[i]
            x = transition_layer(
                x,
                t["bn_gamma"],
                t["bn_beta"],
                t["bn_mean"],
                t["bn_var"],
                t["conv_weight"],
                eps,
            )

    x = F.batch_norm(
        x,
        running_mean=weights["final_bn_mean"],
        running_var=weights["final_bn_var"],
        weight=weights["final_bn_gamma"],
        bias=weights["final_bn_beta"],
        training=False,
        eps=eps,
    )
    x = F.relu(x)
    x = x.mean(dim=(2, 3))
    x = x @ weights["fc_weight"].t() + weights["fc_bias"]
    return x