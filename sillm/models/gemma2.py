from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel
from sillm.models.args import ModelArgs
from sillm.models.gemma import RMSNorm, FeedForward
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
        super().__init__(args)

        self.attn_logit_softcapping = args.attn_logit_softcapping
        self.scale = 1.0 / (args.query_pre_attn_scalar**0.5)

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

        ########
        # Attention softcapping
        # Reference:
        # https://github.com/huggingface/transformers/blob/b7ee1e80b912c6cdd93b54dd77af061fde151d28/src/transformers/models/gemma2/modeling_gemma2.py#L259
        ########
        output = output / self.attn_logit_softcapping
        output = mx.tanh(output)
        output = output * self.attn_logit_softcapping

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.wo(output), (keys, values)

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
        self.final_logit_softcapping = args.final_logit_softcapping
        
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

        ########
        # Final logit softcapping
        # Reference:
        # https://github.com/huggingface/transformers/blob/b7ee1e80b912c6cdd93b54dd77af061fde151d28/src/transformers/models/gemma2/modeling_gemma2.py#L1083
        ########
        out = out / self.args.final_logit_softcapping
        out = mx.tanh(out)
        out = out * self.args.final_logit_softcapping

        return out, cache