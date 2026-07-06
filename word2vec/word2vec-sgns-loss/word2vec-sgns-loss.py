import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor) -> torch.Tensor:
    pos_score = torch.dot(center_vec, pos_vec)
    neg_scores = neg_vecs @ center_vec
    return F.softplus(-pos_score) + F.softplus(neg_scores).sum()