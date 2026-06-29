import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Extract the number of samples (N) and features (D)
    N, D = X.shape
    
    # 1. Initialize weights w as zeros and bias b as 0.0
    w = np.zeros(D)
    b = 0.0
    
    # 2. Gradient Descent Loop
    for _ in range(steps):
        # Calculate the linear combination (z = Xw + b)
        z = np.dot(X, w) + b
        
        # Apply the sigmoid function to get probabilities (p)
        p = _sigmoid(z)
        
        # Calculate the difference between predictions and actual labels
        error = p - y
        
        # Compute gradients (Hint 1)
        # np.dot(X.T, error) acts as X^T(p - y)
        dw = np.dot(X.T, error) / N
        db = np.mean(error)
        
        # Update parameters (Hint 2)
        w = w - lr * dw
        b = b - lr * db
        
    return w, b