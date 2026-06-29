import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    inputs = np.array(x)
    _sigmoid_ = (1 / (1 + np.exp(-inputs)))
    return _sigmoid_
    pass