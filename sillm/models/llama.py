from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel
from sillm.models.args import ModelArgs

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/85dc76f6e0f2cf3ee3d84c211868a6856e163f3f/llms/mlx_lm/models/llama.py#L49
########
class Llama3RoPE(nn.Module):
    def __init__(self,
                 head_dim: int,
                 max_position_embeddings: int = 131072,
                 traditional: bool = True,
                 base: float = 10000,
                 scale: float = 1.0,
                 rope_scaling: dict = None,
                 ):
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional
        self.original_base = base
        self.scale = scale

        ########
        # Calculate base frequencies
        # References:
        # https://github.com/huggingface/transformers/blob/d5a99dfcee6e94065cb7c83cc8ab6fc5daa0cc4e/src/transformers/modeling_rope_utils.py#L318
        ########
        factor = rope_scaling.get("factor", 8.0)
        low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
        old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = self.original_base ** (mx.arange(0, head_dim, 2) / head_dim)
        wavelens = 2 * mx.pi * freqs
        new_base_freqs = []

        smooths = (wavelens - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
        new_base_freqs = freqs * (1 - smooths) * factor + smooths
        new_base_freqs = mx.where(wavelens < high_freq_wavelen, freqs, new_base_freqs)
        new_base_freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, new_base_freqs)

        self.base = new_base_freqs.mean().item()

            
    def __call__(self, x, offset: int = 0):
        seq_len = x.shape[1] + offset
        base = self.base
        if self.max_position_embeddings and seq_len > self.max_position_embeddings:
            base *= (
                (self.scale * seq_len / self.max_position_embeddings) - (self.scale - 1)
            ) ** (self.head_dim / (self.head_dim - 2))

        return mx.fast.rope(
            x,
            self.head_dim,
            traditional=self.traditional,
            base=base,
            scale=self.scale,
            offset=offset,
        )

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

        self.scale = self.args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        rope_type = "default"
        rope_scale = 1.0
        if args.rope_scaling is not None:
            rope_type = (args.rope_scaling.get("type") or args.rope_scaling.get("rope_type") or "default")

            if rope_type == "linear":
                rope_scale = 1 / args.rope_scaling["factor"]

        if rope_type in ("default", "linear"):
            self.rope = nn.RoPE(args.head_dim,
                                traditional=args.rope_traditional,
                                base=args.rope_theta,
                                scale=rope_scale)
        elif rope_type == "llama3":
            self.rope = Llama3RoPE(args.head_dim,
                                   max_position_embeddings=args.max_position_embeddings,
                                   traditional=args.rope_traditional,
                                   base=args.rope_theta,
                                   scale=rope_scale,
                                   rope_scaling=args.rope_scaling)
        else:
            raise NotImplementedError(f"Unknown scaling type {rope_type}")

    def __call__(self,
                 x: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[Tuple[mx.array, mx.array]] = None,
                 ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

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
        self.args = args
        
        self.n_heads = args.n_heads
        self.dim = args.dim
        
        self.attention = Attention(args=args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
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
    Llama model wrapper.
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
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self,
                 inputs: mx.array,
                 cache = None
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
            h, cache[e] = layer.forward(h, mask, cache[e])

        return self.output(self.norm(h)), cache