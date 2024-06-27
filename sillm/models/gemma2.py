from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel
from sillm.models.args import ModelArgs
from sillm.models.gemma import RMSNorm, FeedForward
import sillm.models.llama as llama
    
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
        
        self.attention = llama.Attention(args=args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

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
        h, cache = self.attention(self.attention_norm(x), mask, cache)
        r = self.ffn_norm(h) + x

        h = self.feed_forward(self.pre_feedforward_layernorm(r))
        out = self.post_feedforward_layernorm(h) + r
        
        return out, cache

########
# References:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma.py
########
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
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

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
        h = h * (self.args.dim**0.5)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer.forward(h, mask, cache[e])

        out = self.tok_embeddings.as_linear(self.norm(h))

        return out, cache