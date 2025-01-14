from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel, scaled_dot_product_attention
from sillm.models.args import ModelArgs
from sillm.modules.norm import LayerNorm2D
from sillm.modules.rope import init_rope
import sillm.models.llama as llama
    
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

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.use_qk_norm = args.use_qk_norm

        self.scale = self.args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=args.attention_bias)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.attention_bias)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.attention_bias)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.attention_bias)
        
        if self.use_qk_norm:
            self.q_norm = LayerNorm2D(self.n_heads, args.head_dim, eps=args.norm_eps)
            self.k_norm = LayerNorm2D(self.n_kv_heads, args.head_dim, eps=args.norm_eps)

        self.rope = init_rope(args)

    def __call__(self,
                 x: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[Tuple[mx.array, mx.array]] = None,
                 ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)
        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            if cache.offset > 0 and L > 1:
                mask = BaseModel.create_additive_causal_mask(L, offset=cache.offset, dtype=queries.dtype)
                
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.wo(output)

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
        self.feed_forward = llama.FeedForward(args=args)
        self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=args.norm_bias)

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
        h = self.attention_norm(x)
        a = self.attention(h, mask, cache)
        r = self.feed_forward(h)
        out = a + r + x
        
        return out

class Model(BaseModel):
    """
    Cohere model wrapper.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        # Override RoPE settings
        args.rope_traditional = True

        super().__init__(args)
        self.args = args

        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.logit_scale = args.logit_scale
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=args.norm_bias)

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
            h = layer.forward(h, mask, cache[e])

        out = self.tok_embeddings.as_linear(self.norm(h))
        
        return out * self.logit_scale