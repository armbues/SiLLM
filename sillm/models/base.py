from functools import partial

import mlx.core as mx
import mlx.nn as nn

from sillm.models.args import ModelArgs

class BaseModel(nn.Module):
    """
    Base class for LLM models.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
    
    def __call__(self,
                 inputs: mx.array,
                 cache = None
                 ):
        raise NotImplementedError("Class model.Model is used for inheritance only")
    
    def loss(self,
        inputs: mx.array,
        targets: mx.array,
        lengths: mx.array):
        raise NotImplementedError("Loss function is not implemented for this model")
    
    ########
    # Based on mlx-examples:
    # https://github.com/ml-explore/mlx-examples/blob/047d4650c4f63d55e5bfbaf8f589c1679cbdd971/lora/lora.py#L151
    ########
    def loss(self,
            inputs: mx.array,
            targets: mx.array,
            loss_masks: mx.array
            ):
        """
        Calculate cross-entropy loss.
        Args:
            inputs: Input tokens.
            targets: Target tokens.
            lengths: Lengths of inputs.
        Returns:
            Cross-entropy loss.
        """
        # Run model on inputs
        logits, _ = self.__call__(inputs)
        logits = logits.astype(mx.float32)

        # Calculate the loss
        cross_entropy_loss = nn.losses.cross_entropy(logits, targets) * loss_masks
        num_tokens = loss_masks.sum()
        loss_value = cross_entropy_loss.sum() / num_tokens

        return loss_value, None, num_tokens

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/13794a05da6b4066552abcb25cad44329d96036b/llms/mlx_lm/models/layers.py#L8
########
@partial(mx.compile, shapeless=True)
def rms_norm(x, weight, eps):
    x = x.astype(mx.float32)
    x = x * mx.rsqrt(x.square().mean(-1, keepdims=True) + eps)

    return weight * x.astype(weight.dtype)

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/13794a05da6b4066552abcb25cad44329d96036b/llms/mlx_lm/models/layers.py#L14
########
class RMSNorm(nn.Module):
    """
    Root mean square normalization.
    """
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()

        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return rms_norm(x, self.weight, self.eps)

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/13794a05da6b4066552abcb25cad44329d96036b/llms/mlx_lm/models/layers.py#L25
########
@partial(mx.compile, shapeless=True)
def ln_norm(x, eps, weight=None, bias=None):
    t = x.dtype
    x = x.astype(mx.float32)
    means = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    x = (x - means) * mx.rsqrt(var + eps)
    x = x.astype(t)

    return weight * x + bias if weight is not None else x

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/13794a05da6b4066552abcb25cad44329d96036b/llms/mlx_lm/models/layers.py#L35
########
class LayerNorm(nn.Module):
    """
    Layer normalization.
    """
    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()

        if affine:
            self.bias = mx.zeros((dims,))
            self.weight = mx.ones((dims,))
        self.eps = eps
        self.dims = dims

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}"

    def __call__(self, x: mx.array) -> mx.array:
        if "weight" in self:
            return ln_norm(x, self.eps, self.weight, self.bias)
        else:
            return ln_norm(x, self.eps)