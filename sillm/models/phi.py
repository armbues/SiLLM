import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel
from sillm.core.cache import KVCache
from sillm.models.args import ModelArgs
from sillm.modules.rope import init_rope
from sillm.modules.act import init_act
import sillm.models.llama as llama

class Attention(llama.Attention):
    """
    Multi-head attention module.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        nn.Module.__init__(self)
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=True)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=True)

        rope_dims = int(args.partial_rotary_factor * args.head_dim)
        
        self.rope = nn.RoPE(rope_dims, traditional=False, base=args.rope_theta)
        
    def __call__(self,
                 x: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[KVCache] = None,
                 ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).moveaxis(1, 2)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).moveaxis(1, 2)
        values = values.reshape(B, L, self.n_kv_heads, -1).moveaxis(1, 2)

        if cache is not None:
            if cache.offset > 0 and L > 1:
                mask = BaseModel.create_additive_causal_mask(L, offset=cache.offset)
                
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scale = math.sqrt(1 / queries.shape[-1])
        output = mx.fast.scaled_dot_product_attention(queries.astype(mx.float32), keys, values, scale=scale, mask=mask).astype(values.dtype)
        output = output.moveaxis(2, 1).reshape(B, L, -1)

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

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=True)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=True)
        
        self.act = init_act(args)
    
    def __call__(self,
                 x: mx.array
                 ) -> mx.array:
        """
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        return self.w2(self.act(self.w1(x)))
    
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
        self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = FeedForward(args=args)

    def forward(self,
                x: mx.array,
                mask: Optional[mx.array] = None,
                cache: Optional[KVCache] = None,
                ) -> mx.array:
        h = self.attention_norm(x)
        attn_h = self.attention(h, mask, cache)
        ff_h = self.feed_forward(h)

        return attn_h + ff_h + x
    
class Model(llama.Model):
    """
    Phi model wrapper.
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
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=True)