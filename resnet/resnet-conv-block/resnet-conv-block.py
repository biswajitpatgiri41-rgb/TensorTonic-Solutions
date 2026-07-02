import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    
    
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    Ws = np.array(Ws)
    
    
    def relu(a):
        return np.maximum(0, a)
    
    
    s = x @ Ws
    
    # Main path
    h = relu(x @ W1)
    z = h @ W2          
    
    
    y = relu(z + s)
    
    return y