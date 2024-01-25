from typing import Optional, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from sillm.model import BaseModel
from sillm.args import ModelArgs
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

    def __call__(self,
                 x,
                 offset: int = 0
                 ):
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
    def __init__(self, args: ModelArgs):
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

    def __call__(self,
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
# https://github.com/ml-explore/mlx-examples/blob/1415595409874971b5ad96d55980e7d4aa8a8043/lora/models/mixtral.py#L141
########
class FeedForward(nn.Module):
    """
    MoE Feed-forward module for Mixtral models.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.num_experts = args.moe["num_experts"]
        self.num_experts_per_tok = args.moe["num_experts_per_tok"]

        self.experts = [llama.FeedForward(args) for _ in range(self.num_experts)]
        self.gate = nn.Linear(args.dim, self.num_experts, bias=False)

    def __call__(self,
                 x: mx.array,
                 training: bool = False
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
        
        expert_indices = mx.stop_gradient(mx.argpartition(-gate_logits, kth=top_k, axis=-1)[:, :top_k])
        expert_scores = mx.softmax(mx.take_along_axis(gate_logits, expert_indices, axis=-1).astype(mx.float32), axis=-1)            
        mx.eval(expert_indices)
        expert_indices = np.array(expert_indices)

        y = mx.zeros((x.shape[0], self.num_experts_per_tok, x.shape[-1]))

        for e, expert in enumerate(self.experts):
            idx1, idx2 = map(mx.array, np.where(expert_indices == e))
            if idx1.size == 0:
                continue
            y[idx1, idx2] = expert(x[idx1])

        y = (y * expert_scores[:, :, None]).sum(axis=1)

        if training:
            ########
            # Calculate expert router loss
            # References:
            # https://github.com/huggingface/transformers/blob/19e83d174c1e2802a459c9b5831628817e1c286f/src/transformers/models/mixtral/modeling_mixtral.py#L77
            # https://arxiv.org/abs/2101.03961 (Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity)
            ########
            
            # Calculate routing weights and router probability per expert
            routing_weights = mx.softmax(gate_logits, axis=-1)
            router_prob_per_expert = mx.mean(routing_weights, axis=0)
            
            # Calculate expert counts and tokens per expert
            expert_counts = np.bincount(expert_indices.flatten(), minlength=8)
            tokens_per_expert = expert_counts / expert_counts.sum()
            
            # Calculate expert loss and total router loss
            expert_losses = tokens_per_expert * router_prob_per_expert
            router_loss = expert_losses.sum()

            return y.reshape(orig_shape), router_loss

        return y.reshape(orig_shape), 0.0
    
########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/d8680a89f986492dbc27c36af3294034db26458f/llms/mixtral/mixtral.py#L163
########
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = llama.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = llama.RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(self,
                 x: mx.array,
                 mask: mx.array = None,
                 cache = None,
                 training: bool = False
                 ) -> mx.array:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r, router_loss = self.feed_forward(self.ffn_norm(h), training)
        out = h + r
        return out, cache, router_loss
    
########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/llms/mixtral/mixtral.py#L187
########
class Model(BaseModel):
    """
    Mixtral model.
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
        self.norm = llama.RMSNorm(args.dim, eps=args.norm_eps)
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
            h, cache[e], _ = layer(h, mask, cache[e])

        return self.output(self.norm(h[:, T - 1 : T, :])), cache
    
    def loss(self,
        inputs: mx.array,
        targets: mx.array,
        lengths: mx.array):
        """
        Calculate loss for inputs.
        Args:
            inputs: Input tokens.
            targets: Target tokens.
            lengths: Lengths of inputs.
        Returns:
            Cross-entropy + router loss.
        """
        h = self.tok_embeddings(inputs)

        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        aux_loss = 0.0
        for layer in self.layers:
            h, _, router_loss = layer(h, mask, None, training=True)
            aux_loss += router_loss

        logits = self.output(self.norm(h[:, T - 1 : T, :])).astype(mx.float32)

        # Mask padding tokens
        length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]
        cross_entropy_loss = nn.losses.cross_entropy(logits, targets) * length_mask
        num_tokens = length_mask.sum()
        cross_entropy_loss = cross_entropy_loss.sum() / num_tokens

        overall_loss = cross_entropy_loss + aux_loss * self.router_aux_loss_coef

        return overall_loss, num_tokens