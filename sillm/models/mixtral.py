from typing import Optional, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel
from sillm.models.args import ModelArgs
from sillm.modules.switch import SwitchGLU
import sillm.models.llama as llama

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/f530f56df2738a54982c4541189a8c8d7cd94c44/llms/mlx_lm/models/mixtral.py#L97
########
class FeedForward(nn.Module):
    """
    MoE Feed-forward module for Mixtral models.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.num_experts = args.moe["num_experts"]
        self.num_experts_per_tok = args.moe["num_experts_per_tok"]

        self.gate = nn.Linear(args.dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(args.dim, args.hidden_dim, self.num_experts, bias=False)

    def __call__(self,
                 x: mx.array
                 ) -> mx.array:
        """
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        gates = self.gate(x)

        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        return y
    
########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/d8680a89f986492dbc27c36af3294034db26458f/llms/mixtral/mixtral.py#L163
########
class TransformerBlock(nn.Module):
    """
    Transformer block for Mixtral models.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = llama.Attention(args=args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(self,
                 x: mx.array,
                 mask: mx.array = None,
                 cache = None
                 ) -> mx.array:
        r = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r

        return out
    
########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/llms/mixtral/mixtral.py#L187
########
class Model(BaseModel):
    """
    Mixtral model wrapper.
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
        self.router_aux_loss_coef = args.router_aux_loss_coef

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
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h = layer.forward(h, mask, cache[e])

        return self.output(self.norm(h))
    
    def preprocess_weights(self,
                           weights: dict
                           ) -> dict:
        num_experts = self.args.moe["num_experts"]
        
        for l in range(self.args.n_layers):
            for k in ("w1", "w2", "w3"):
                for n in ("weight", "scales", "biases"):
                    prefix = f"layers.{l}.feed_forward"

                    if f"{prefix}.experts.0.{k}.{n}" not in weights:
                        continue

                    experts_keys = [prefix + f".experts.{i}.{k}.{n}" for i in range(num_experts)]
                    weights[prefix + f".switch_mlp.{k}.{n}"] = mx.stack([weights[key] for key in experts_keys])

                    for key in experts_keys:
                        del weights[key]

        return weights