from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

import sillm.model as model
import sillm.modules as modules

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/047d4650c4f63d55e5bfbaf8f589c1679cbdd971/llms/mixtral/mixtral.py#L43
########
class RoPE(nn.RoPE):
    """
    Rotary Position Embedding module for Mixtral models.
    """
    def __init__(self,
                 dims: int,
                 traditional: bool = False,
                 base: float = 1000000):
        """
        Args:
            dims: Embedding dimensions.
            traditional: Whether to use traditional RoPE.
            base: Base for traditional RoPE.
        """
        super().__init__(dims, traditional)

        self.base = base

    def __call__(self, x, offset: int = 0):
        """
        Args:
            x: Input tensor.
            offset: Offset for RoPE.
        Returns:
            Output tensor.
        """
        shape = x.shape
        x = mx.reshape(x, (-1, shape[-2], shape[-1]))
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, base=self.base, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return mx.reshape(rx, shape)

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/047d4650c4f63d55e5bfbaf8f589c1679cbdd971/llms/mixtral/mixtral.py#L132
########
class FeedForward(nn.Module):
    """
    Feed-forward module for Mixtral models.
    """
    def __init__(self, args: model.ModelArgs):
        super().__init__()

        self.num_experts = args.moe["num_experts"]
        self.num_experts_per_tok = args.moe["num_experts_per_tok"]
        self.experts = [modules.FeedForward(args) for _ in range(self.num_experts)]
        self.gate = nn.Linear(args.dim, self.num_experts, bias=False)

    def __call__(self, x) -> mx.array:
        """
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.gate(x)
        inds = mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne]
        scores = mx.softmax(
            mx.take_along_axis(gates, inds, axis=-1).astype(mx.float32),
            axis=-1,
        ).astype(gates.dtype)

        y = []
        for xt, st, it in zip(x, scores, inds.tolist()):
            yt = mx.concatenate([self.experts[e](xt)[:, None] for e in it], axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt[None, :])
        y = mx.concatenate(y)

        return y.reshape(orig_shape)
    
########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/llms/mixtral/mixtral.py#L187
########
class Model(model.Model):
    """
    Mixtral model.
    """
    def __init__(self, args: model.ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        super().__init__(args)
        self.args = args

        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [modules.TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = modules.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
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
            h, cache[e] = layer(h, mask, cache[e])

        return self.output(self.norm(h[:, T - 1 : T, :])), cache
    
########
# See mixtral transformers implementation:
# https://github.com/huggingface/transformers/blob/19e83d174c1e2802a459c9b5831628817e1c286f/src/transformers/models/mixtral/modeling_mixtral.py#L77
########
def load_balancing_loss(
        gate_logits: mx.array,
        num_experts: int,
        top_k: int = 2
        ):
    """
    Calculate load balancing loss.
    Args:
        gate_logits: Gate logits.
        num_experts: Total number of experts in model.
        top_k: Number of experts to consider.
    """
    # Calculate routing weights
    routing_weights = mx.softmax(gate_logits, axis=-1)

    # Calculate selected experts and their probabilities
    selected_experts = mx.topk(routing_weights, k=top_k, axis=-1)

    # Calculate expert mask
    expert_mask = mx.one_hot(selected_experts, num_experts)

    # Calculate tokens per expert
    tokens_per_expert = mx.mean(expert_mask, axis=1)

    # Calculate router probability per expert
    router_prob_per_expert = mx.mean(routing_weights, axis=0)

    # Calculate overall loss
    overall_loss = mx.sum(tokens_per_expert * router_prob_per_expert)

    return overall_loss * num_experts