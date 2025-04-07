from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel, scaled_dot_product_attention
from sillm.core.cache import KVCache
from sillm.models.args import ModelArgs
from sillm.modules.norm import RMSNorm
import sillm.models.llama as llama

class Attention(nn.Module):
    """
    Multi-head attention module.
    """
    def __init__(self, args: ModelArgs, layer_index: int):
        """
        Args:
            args: Model arguments.
        """
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.scale = self.args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=args.attention_bias)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.attention_bias)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.attention_bias)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.attention_bias)

        self.q_norm = RMSNorm(dims=args.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(dims=args.head_dim, eps=args.rms_norm_eps)

        layer_sliding = (layer_index + 1) % self.args.sliding_window_pattern != 0
        if layer_sliding:
            rope_theta = args.rope_local_base_freq
        else:
            rope_theta = args.rope_global_base_freq
        self.rope = nn.RoPE(args.head_dim, traditional=args.rope_traditional, base=rope_theta)

    def __call__(self,
                 x: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[KVCache] = None,
                 ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if isinstance(mask, mx.array) and mask.shape[-1] != keys.shape[-2]:
            mask = mask[..., -keys.shape[-2]:]

        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.wo(output)
    
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

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=args.mlp_bias)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=args.mlp_bias)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=args.mlp_bias)

    def __call__(self, x) -> mx.array:
        """
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        return self.w2(nn.gelu_approx(self.w1(x)) * self.w3(x))

class TransformerBlock(llama.TransformerBlock):
    """
    Transformer block.
    """
    def __init__(self, args: ModelArgs, layer_index: int):
        """
        Args:
            args: Model arguments.
        """
        nn.Module.__init__(self)
        self.args = args

        self.n_heads = args.n_heads
        self.dim = args.dim
        
        self.attention = Attention(args=args, layer_index=layer_index)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[KVCache] = None,
            ) -> mx.array:
        """
        Args:
            x: Input tensor.
            mask: Mask tensor.
            cache: Cache from previous forward pass.
        Returns:
            Output tensor and cache.
        """
        h = self.attention(self.attention_norm(x), mask, cache)
        r = self.ffn_norm(h) + x

        h = self.feed_forward(self.pre_feedforward_layernorm(r))
        out = self.post_feedforward_layernorm(h) + r
        
        return out

class Model(llama.Model):
    """
    Gemma model wrapper.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        BaseModel.__init__(self, args)
        self.args = args

        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.sliding_window_pattern = args.sliding_window_pattern
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args, layer_index=i) for i in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.embed_scale = None

        self.output = None

    def __call__(self,
                 inputs: mx.array,
                 cache = None
                 ):
        """
        Args:
            inputs: Input tokens.
            cache: Cache from previous forward pass.
        Returns:
            Output logits.
        """
        h = self.tok_embeddings(inputs)

        if self.embed_scale is None:
            self.embed_scale = mx.array(self.args.hidden_dim**0.5, mx.bfloat16).astype(h.dtype)
        h *= self.embed_scale

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = BaseModel.create_attention_mask(h, cache[self.sliding_window_pattern - 1 : self.sliding_window_pattern])
        sliding_window_mask = BaseModel.create_attention_mask(h, cache)

        for i, layer in enumerate(self.layers):
            layer_sliding = (i + 1) % self.args.sliding_window_pattern != 0
            mask = sliding_window_mask if layer_sliding else full_mask

            h = layer.forward(h, mask, cache[i])

        return self.tok_embeddings.as_linear(self.norm(h))