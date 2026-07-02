import numpy as np

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        x = np.array(x)
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            x_hat = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta

def relu(x):
    return np.maximum(0, x)

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode="post"):
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    gamma1 = np.array(gamma1)
    beta1 = np.array(beta1)
    gamma2 = np.array(gamma2)
    beta2 = np.array(beta2)

    bn1 = BatchNorm(W1.shape[1])
    bn2 = BatchNorm(W2.shape[1])
    bn1.gamma = gamma1
    bn1.beta = beta1
    bn2.gamma = gamma2
    bn2.beta = beta2

    if mode == "post":
        out = x @ W1
        out = bn1.forward(out, training=True)
        out = relu(out)
        out = out @ W2
        out = bn2.forward(out, training=True)
        out = out + x
        out = relu(out)
    else:
        out = bn1.forward(x, training=True)
        out = relu(out)
        out = out @ W1
        out = bn2.forward(out, training=True)
        out = relu(out)
        out = out @ W2
        out = out + x

    return {"output": out.tolist(), "mode": mode}