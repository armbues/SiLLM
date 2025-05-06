from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel, scaled_dot_product_attention
from sillm.core.cache import KVCache
from sillm.models.args import ModelArgs
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
        nn.Module.__init__(self)
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.scale = self.args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        self.q_norm = nn.RMSNorm(dims=args.head_dim, eps=args.norm_eps)
        self.k_norm = nn.RMSNorm(dims=args.head_dim, eps=args.norm_eps)

        self.rope = init_rope(args)
        
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

        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.wo(output)

class TransformerBlock(llama.TransformerBlock):
    """
    Transformer block.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        nn.Module.__init__(self)
        self.args = args

        self.n_heads = args.n_heads
        self.dim = args.dim
        
        self.attention = Attention(args=args)
        self.feed_forward = llama.FeedForward(args=args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

########
# References:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
########
class Model(llama.Model):
    """
    Qwen 3 model wrapper.
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
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

        if args.tie_word_embeddings:
            self.output = None
        else:
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

        mask = BaseModel.create_attention_mask(h, cache)
        
        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h = layer.forward(h, mask, cache[e])

        if self.output is None:
            out = self.tok_embeddings.as_linear(self.norm(h))
        else:
            out = self.output(self.norm(h))

        return out