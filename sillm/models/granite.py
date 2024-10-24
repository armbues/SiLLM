from typing import Optional

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

        self.scale = args.attention_multiplier

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=args.attention_bias)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.attention_bias)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.attention_bias)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.attention_bias)

        self.rope = init_rope(args)

class TransformerBlock(nn.Module):
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

        self.residual_multiplier = args.residual_multiplier
        
        self.attention = Attention(args=args)
        self.feed_forward = llama.FeedForward(args=args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

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
        h = x + self.attention(self.attention_norm(x), mask, cache) * self.residual_multiplier
        out = h + self.feed_forward(self.ffn_norm(h)) * self.residual_multiplier
        
        return out

########
# References:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite/modeling_granite.py
########
class Model(llama.Model):
    """
    Granite model wrapper.
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

        self.embedding_multiplier = args.embedding_multiplier
        self.logits_scaling = args.logits_scaling
        
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
            Output logits.
        """
        h = self.tok_embeddings(inputs) * self.embedding_multiplier

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1]).astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h = layer.forward(h, mask, cache[e])

        if self.output is None:
            logits = self.tok_embeddings.as_linear(self.norm(h))
        else:
            logits = self.output(self.norm(h))

        logits = logits / self.logits_scaling

        return logits