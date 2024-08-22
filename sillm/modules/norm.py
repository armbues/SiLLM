import mlx.core as mx
import mlx.nn as nn

class LayerNorm2D(nn.Module):
    def __init__(self, d1, d2, eps):
        super().__init__()

        self.weight = mx.zeros((d1, d2))
        self.eps = eps

    def __call__(self, x):
        return self.weight * mx.fast.layer_norm(x, None, None, self.eps)
    
class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization module.
    """
    def __init__(self,
                 dims: int,
                 eps: float = 1e-6):
        super().__init__()

        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)