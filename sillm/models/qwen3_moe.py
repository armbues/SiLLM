from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel, scaled_dot_product_attention
from sillm.core.cache import KVCache
from sillm.models.args import ModelArgs
from sillm.modules.rope import init_rope
from sillm.modules.switch import SwitchGLU
import sillm.models.llama as llama
import sillm.models.qwen3 as qwen3

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        super().__init__()

        self.num_experts_per_tok = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(args.dim, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(args.dim, args.moe_intermediate_size, args.num_experts, bias=False)

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
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        return y

class TransformerBlock(llama.TransformerBlock):
    """
    Transformer block.
    """
    def __init__(self, args: ModelArgs, layer_index: int):
        """
        Args:
            args: Model arguments.
        """
        nn.Module.__init__(self)
        self.args = args

        self.n_heads = args.n_heads
        self.dim = args.dim
        
        self.attention = qwen3.Attention(args=args)
        self.feed_forward = FeedForward(args=args)

        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

########
# References:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
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
        self.layers = [TransformerBlock(args=args, layer_index=i) for i in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

        if args.tie_word_embeddings:
            self.output = None
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def preprocess_weights(self,
                           weights: dict
                           ) -> dict:
        num_experts = self.args.num_experts
        
        key_map = {
            "gate_proj": "w1",
            "down_proj": "w2",
            "up_proj": "w3",
        }
        for l in range(self.args.n_layers):
            for k in ("gate_proj", "down_proj", "up_proj"):
                for n in ("weight", "scales", "biases"):
                    prefix = f"layers.{l}"

                    if f"{prefix}.mlp.experts.0.{k}.{n}" not in weights:
                        continue

                    experts_keys = [prefix + f".mlp.experts.{i}.{k}.{n}" for i in range(num_experts)]
                    weights[prefix + f".feed_forward.switch_mlp.{key_map[k]}.{n}"] = mx.stack([weights[key] for key in experts_keys])

                    for key in experts_keys:
                        del weights[key]

        return weights