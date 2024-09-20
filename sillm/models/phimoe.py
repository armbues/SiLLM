from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel
from sillm.models.args import ModelArgs
from sillm.modules.rope import init_rope
from sillm.modules.switch import SwitchGLU
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

    def preprocess_weights(self,
                           weights: dict
                           ) -> dict:
        num_experts = self.args.num_local_experts
        
        for l in range(self.args.n_layers):
            for k in ("w1", "w2", "w3"):
                prefix = f"layers.{l}.feed_forward"

                experts_keys = [prefix + f".experts.{i}.{k}.weight" for i in range(num_experts)]
                weights[prefix + f".switch_mlp.{k}.weight"] = mx.stack([weights[key] for key in experts_keys])

                for key in experts_keys:
                    del weights[key]

        return weights