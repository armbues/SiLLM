from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from sillm.model import BaseModel
from sillm.args import ModelArgs

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/047d4650c4f63d55e5bfbaf8f589c1679cbdd971/llms/llama/llama.py#L31
########
class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization module.
    """
    def __init__(self,
                 dims: int,
                 eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/047d4650c4f63d55e5bfbaf8f589c1679cbdd971/lora/models.py#L151
########
class Attention(nn.Module):
    """
    Multi-head attention module.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        self.rope = nn.RoPE(args.head_dim, traditional=True, base=args.rope_theta)
        # TODO RoPE scaling

    def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)

            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.wo(output), (keys, values)

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/047d4650c4f63d55e5bfbaf8f589c1679cbdd971/llms/llama/llama.py#L104
########
class FeedForward(nn.Module):
    """
    Feed-forward module.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        """
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))
    
########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/047d4650c4f63d55e5bfbaf8f589c1679cbdd971/llms/llama/llama.py#L116
########
class TransformerBlock(nn.Module):
    """
    Transformer block.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Args:
            x: Input tensor.
            mask: Mask tensor.
            cache: Cache from previous forward pass.
        Returns:
            Output tensor and cache.
        """
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        
        return out, cache

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/047d4650c4f63d55e5bfbaf8f589c1679cbdd971/llms/llama/llama.py#L140
########
class Model(BaseModel):
    """
    Llama model.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        super().__init__(args)
        self.args = args

        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self,
                 inputs: mx.array,
                 cache=None
                 ):
        """
        Args:
            inputs: Input tokens.
            cache: Cache from previous forward pass.
        Returns:
            Output logits and cache.
        """
        h = self.tok_embeddings(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.output(self.norm(h)), cache
    
    ########
    # Based on mlx-examples:
    # https://github.com/ml-explore/mlx-examples/blob/047d4650c4f63d55e5bfbaf8f589c1679cbdd971/lora/lora.py#L151
    ########
    def loss(self,
            inputs: mx.array,
            targets: mx.array,
            lengths: mx.array):
        """
        Calculate loss for inputs.
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

        # Mask padding tokens
        length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

        # Calculate the loss
        cross_entropy_loss = nn.losses.cross_entropy(logits, targets) * length_mask
        num_tokens = length_mask.sum()
        cross_entropy_loss = cross_entropy_loss.sum() / num_tokens

        return cross_entropy_loss, num_tokens