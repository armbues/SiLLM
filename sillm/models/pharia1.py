from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from sillm.core.cache import KVCache
from sillm.models.base import BaseModel
from sillm.models.args import ModelArgs
from sillm.modules.act import init_act
import sillm.models.llama as llama

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

        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=True)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=True)
        
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
        return self.w2(self.act(self.w3(x)))
    
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
        
        self.attention = llama.Attention(args=args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=args.mlp_bias)
        self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=args.mlp_bias)

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
        r = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        
        return out

class Model(llama.Model):
    """
    Pharia-1 model wrapper.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        # Fix Pharia-1 model arguments
        args.rope_traditional = True

        BaseModel.__init__(self, args)
        self.args = args

        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)