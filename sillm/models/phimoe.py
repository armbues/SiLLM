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

        self.scale = self.args.head_dim ** -0.5
        
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=True)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=True)
        
        self.rope = init_rope(args)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        super().__init__()

        self.num_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok

        self.gate = nn.Linear(args.dim, self.num_experts, bias=False)
        self.experts = [llama.FeedForward(args=args) for _ in range(self.num_experts)]

    def __call__(self,
                 x: mx.array
                 ) -> mx.array:
        """
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        top_k = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gate_logits = self.gate(x)

        expert_indices = mx.stop_gradient(mx.argpartition(-gate_logits, kth=top_k-1, axis=-1)[..., :top_k])
        expert_scores = mx.take_along_axis(gate_logits, expert_indices, axis=-1)
        expert_scores = mx.softmax(expert_scores, axis=-1, precise=True)

        y = []
        for xt, st, it in zip(x, expert_scores, expert_indices.tolist()):
            yt = mx.concatenate([self.experts[e](xt)[:, None] for e in it], axis=-1)
            yt = (yt * st).sum(axis=-1)

            y.append(yt[None, :])
        y = mx.concatenate(y)

        return y.reshape(orig_shape)

class TransformerBlock(llama.TransformerBlock):
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
        self.attention_norm = nn.LayerNorm(args.dim, eps=args.rms_norm_eps)
        self.feed_forward = FeedForward(args)
        self.ffn_norm = nn.LayerNorm(args.dim, eps=args.rms_norm_eps)

class Model(llama.Model):
    """
    Phi-MoE model wrapper.
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
        self.norm = nn.LayerNorm(args.dim, eps=args.rms_norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=True)