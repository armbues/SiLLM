from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

import sillm.model as model
import sillm.llama as llama

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
# https://github.com/ml-explore/mlx-examples/blob/d8680a89f986492dbc27c36af3294034db26458f/llms/mixtral/mixtral.py#L63
########
class Attention(nn.Module):
    """
    Multi-head attention module.
    """
    def __init__(self, args: model.ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = RoPE(args.head_dim, traditional=True, base=args.rope_theta)
        # TODO RoPE scaling

    def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)

            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.wo(output), (keys, values)

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
        self.experts = [llama.FeedForward(args) for _ in range(self.num_experts)]
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
        print(gates)

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
# https://github.com/ml-explore/mlx-examples/blob/d8680a89f986492dbc27c36af3294034db26458f/llms/mixtral/mixtral.py#L163
########
class TransformerBlock(nn.Module):
    def __init__(self, args: model.ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = llama.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = llama.RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache
    
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
        self.layers = [llama.TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = llama.RMSNorm(args.dim, eps=args.norm_eps)
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
    
def one_hot(
        indices: mx.array,
        num_classes: int
        ):
    encoded = [[0] * num_classes for _ in range(len(indices))]
    for i, index in enumerate(indices):
        encoded[i][index] = 1

    return encoded

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
    expert_mask = one_hot(selected_experts, num_experts)

    # Calculate tokens per expert
    tokens_per_expert = mx.mean(expert_mask, axis=1)

    # Calculate router probability per expert
    router_prob_per_expert = mx.mean(routing_weights, axis=0)

    # Calculate overall loss
    overall_loss = mx.sum(tokens_per_expert * router_prob_per_expert)

    return overall_loss * num_experts