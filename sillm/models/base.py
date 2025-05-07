import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from sillm.models.args import ModelArgs
from sillm.core.cache import KVCache, QuantizedKVCache

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/32d10036de94af07733c247ca44702e8135d068a/llms/mlx_lm/models/base.py#L26
########
def create_bool_causal_mask(N: int, offset: int = 0):
    rinds = mx.arange(offset + N)[None]
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    mask = linds >= rinds

    return mask

def create_additive_causal_mask(N: int, offset: int = 0, dtype: mx.Dtype = mx.float32):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    mask = mask.astype(dtype) * mx.finfo(dtype).min
    
    return mask

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
        logits = self.__call__(inputs)
        logits = logits.astype(mx.float32)

        # Calculate the loss
        cross_entropy_loss = nn.losses.cross_entropy(logits, targets) * loss_masks
        num_tokens = loss_masks.sum()
        loss_value = cross_entropy_loss.sum() / num_tokens

        return loss_value, None, num_tokens
    
    @staticmethod
    def create_attention_mask(h: mx.array,
                              cache = None
                              ):
        """
        Create attention mask.
        Args:
            h: Input tensor.
            cache: Cache from previous forward pass.
        Returns:
            Attention mask.
        """
        L = h.shape[1]
        
        mask = None
        if L > 1:
            if cache is not None and cache[0] is not None:
                mask = create_bool_causal_mask(L, cache[0].offset)
            else:
                mask = create_bool_causal_mask(L)

        return mask
    
########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/dfa4dd6c93c4c2f81bfed6becb8af5cc3a89ae61/llms/mlx_lm/models/base.py#L56
########
def quantized_scaled_dot_product_attention(queries: mx.array,
                                           keys: mx.array,
                                           values: mx.array,
                                           cache: QuantizedKVCache = None,
                                           scale: float = 1.0,
                                           mask: mx.array = None
                                           ) -> mx.array:
    """
    Scaled dot-product attention with quantized KV cache.
    """
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads
    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), keys)
        values = tree_map(lambda x: mx.expand_dims(x, axis=-3), values)
    scores = mx.quantized_matmul(queries, *keys, transpose=True, group_size=cache.group_size, bits=cache.bits)

    if mask is not None:
        scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(scores, *values, transpose=False, group_size=cache.group_size, bits=cache.bits)
    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out

def scaled_dot_product_attention(queries: mx.array,
                                 keys: mx.array,
                                 values: mx.array,
                                 cache: KVCache = None,
                                 scale: float = 1.0,
                                 mask: mx.array = None
                                 ) -> mx.array:
    """
    Scaled dot-product attention that transparently handles quantized KV cache.
    """
    if isinstance(cache, QuantizedKVCache):
        return quantized_scaled_dot_product_attention(queries, keys, values, cache=cache, scale=scale, mask=mask)
    else:
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask)