import numpy as np

class GAN:
    def __init__(self, a, b):
        if isinstance(a, (list, np.ndarray)):
            self.G_W = np.asarray(a, dtype=float)
            self.D_W = np.asarray(b, dtype=float)
            self.noise_dim, self.data_dim = self.G_W.shape
        else:
            self.data_dim = int(a)
            self.noise_dim = int(b)
            self.G_W = np.random.randn(self.noise_dim, self.data_dim) * 0.1
            self.D_W = np.random.randn(self.data_dim, 1) * 0.1

    def generate(self, x):
        if np.isscalar(x):
            z = np.random.randn(int(x), self.noise_dim)
        else:
            z = np.asarray(x, dtype=float)
        return np.tanh(np.dot(z, self.G_W))

    def discriminate(self, x):
        x = np.asarray(x, dtype=float)
        logits = np.dot(x, self.D_W)
        return 1.0 / (1.0 + np.exp(-logits))

    def train_step(self, real_data, z=None):
        real_data = np.asarray(real_data, dtype=float)

        if z is None:
            z = np.random.randn(real_data.shape[0], self.noise_dim)
        else:
            z = np.asarray(z, dtype=float)

        fake_data = self.generate(z)

        real_probs = np.clip(self.discriminate(real_data), 1e-8, 1 - 1e-8)
        fake_probs = np.clip(self.discriminate(fake_data), 1e-8, 1 - 1e-8)

        d_loss = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
        g_loss = -np.mean(np.log(fake_probs))

        return {
            "d_loss": round(float(d_loss), 4),
            "g_loss": round(float(g_loss), 4)
        }