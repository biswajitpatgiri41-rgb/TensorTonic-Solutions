import numpy as np


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        hidden_dim = int(mlp_ratio * embed_dim)

        
        if W_patch is not None:
            self.W_patch = np.array(W_patch, dtype=float)
        else:
            patch_dim = patch_size * patch_size * 3
            self.W_patch = np.random.randn(patch_dim, embed_dim) * 0.02

       
        if cls_token is not None:
            self.cls_token = np.array(cls_token, dtype=float)
        else:
            self.cls_token = np.random.randn(1, 1, embed_dim) * 0.02

        
        if pos_embed is not None:
            self.pos_embed = np.array(pos_embed, dtype=float)
        else:
            self.pos_embed = np.random.randn(1, self.num_patches + 1, embed_dim) * 0.02

        
        if encoder_weights is not None:
            self.encoder_weights = []
            for block in encoder_weights:
                b = {}
                for k, v in block.items():
                    b[k] = np.array(v, dtype=float)
                self.encoder_weights.append(b)
        else:
            self.encoder_weights = []
            for _ in range(depth):
                block = {
                    "Wq": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wk": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wv": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wo": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "W1": np.random.randn(embed_dim, hidden_dim) * 0.02,
                    "W2": np.random.randn(hidden_dim, embed_dim) * 0.02,
                }
                self.encoder_weights.append(block)

        
        if W_head is not None:
            self.W_head = np.array(W_head, dtype=float)
        else:
            self.W_head = np.random.randn(embed_dim, num_classes) * 0.02

    def _patch_embed(self, x):
        """Split image (B, H, W, C) into patches and project to embed_dim."""
        B, H, W, C = x.shape
        P = self.patch_size
        h = H // P
        w = W // P
        
        patches = (
            x.reshape(B, h, P, w, P, C)
             .transpose(0, 1, 3, 2, 4, 5)
             .reshape(B, h * w, P * P * C)
        )
        return patches @ self.W_patch  

    def _msa(self, x, block):
        
        B, N, D = x.shape
        H = self.num_heads
        d = D // H

        Q = (x @ block["Wq"]).reshape(B, N, H, d).transpose(0, 2, 1, 3)
        K = (x @ block["Wk"]).reshape(B, N, H, d).transpose(0, 2, 1, 3)
        V = (x @ block["Wv"]).reshape(B, N, H, d).transpose(0, 2, 1, 3)

        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d)
        attn = softmax(scores, axis=-1)

        out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, N, D)
        return out @ block["Wo"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        
        B = x.shape[0]

        
        z = self._patch_embed(x)                          

       
        cls_tokens = np.tile(self.cls_token, (B, 1, 1))   
        z = np.concatenate([cls_tokens, z], axis=1)        

        
        z = z + self.pos_embed                             

        
        for block in self.encoder_weights:
            
            z_ln = layer_norm(z)
            z = z + self._msa(z_ln, block)

            
            z_ln = layer_norm(z)
            z = z + (gelu(z_ln @ block["W1"]) @ block["W2"])

        
        cls_out = layer_norm(z[:, 0, :])                   
        logits = cls_out @ self.W_head                     

        return logits