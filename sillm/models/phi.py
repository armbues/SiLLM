from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel
from sillm.core.cache import KVCache
from sillm.models.args import ModelArgs
from sillm.modules.rope import init_rope
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

        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = self.args.head_dim ** -0.5
        self.rope_dims = int(args.partial_rotary_factor * args.head_dim)

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=True)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=True)
        
        self.rope = init_rope(args)
        
    def __call__(self,
                 x: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[KVCache] = None,
                 ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if self.repeats > 1:
            keys = mx.repeat(keys, self.repeats, axis=1)
            values = mx.repeat(values, self.repeats, axis=1)

        if cache is not None:
            if cache.offset > 0 and L > 1:
                mask = BaseModel.create_additive_causal_mask(L, offset=cache.offset)
                
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries.astype(mx.float32)
        keys = keys.astype(mx.float32)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

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
    
    def __call__(self,
                 x: mx.array
                 ) -> mx.array:
        """
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        return self.w2(nn.gelu_approx(self.w1(x)))
    
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
                cache: Optional[Tuple[mx.array, mx.array]] = None,
                ) -> mx.array:
        h = self.attention_norm(x)
        attn_h, cache = self.attention(h, mask, cache)
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