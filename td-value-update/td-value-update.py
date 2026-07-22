import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    V_new = V.copy()
    delta = r + gamma * V[s_next] - V[s]
    V_new[s] = V[s] + alpha * delta
    return V_new