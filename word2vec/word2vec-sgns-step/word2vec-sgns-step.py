import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    W_in_updated = W_in.clone()
    W_out_updated = W_out.clone()

    v_c = W_in[center_id].clone()
    u_o = W_out[pos_id].clone()
    u_negs = W_out[neg_ids].clone()

    score_o = torch.dot(v_c, u_o)
    coeff_o = torch.sigmoid(score_o) - 1.0

    neg_scores = u_negs @ v_c
    coeff_negs = torch.sigmoid(neg_scores)

    grad_v = coeff_o * u_o + (coeff_negs.unsqueeze(1) * u_negs).sum(dim=0)
    W_in_updated[center_id] -= lr * grad_v

    W_out_updated[pos_id] -= lr * (coeff_o * v_c)

    for i, neg_id in enumerate(neg_ids):
        W_out_updated[neg_id] -= lr * (coeff_negs[i] * v_c)

    return W_in_updated, W_out_updated