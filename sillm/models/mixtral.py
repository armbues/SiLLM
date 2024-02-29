from typing import Optional, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel
from sillm.models.args import ModelArgs
import sillm.models.llama as llama

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

        self.gate = nn.Linear(args.dim, self.num_experts, bias=False)
        self.experts = [llama.FeedForward(args=args) for _ in range(self.num_experts)]

    def __call__(self,
                 x: mx.array,
                 training_loss: bool = False
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
        expert_scores = mx.softmax(mx.take_along_axis(gate_logits, expert_indices, axis=-1).astype(mx.float32), axis=-1).astype(gate_logits.dtype)
        
        if self.training:
            y = mx.zeros((x.shape[0], self.num_experts_per_tok, x.shape[-1]))
            for e, expert in enumerate(self.experts):
                idx1, idx2 = map(mx.array, np.where(expert_indices == e))
                if idx1.size == 0:
                    continue
                y[idx1, idx2] = expert(x[idx1])

            y = (y * expert_scores[:, :, None]).sum(axis=1)
        else:
            y = []
            for xt, st, it in zip(x, expert_scores, expert_indices.tolist()):
                yt = mx.concatenate([self.experts[e](xt)[:, None] for e in it], axis=-1)
                yt = (yt * st).sum(axis=-1)
                y.append(yt[None, :])
            y = mx.concatenate(y)

        if training_loss:
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
        else:
            router_loss = 0.0

        return y.reshape(orig_shape), router_loss
    
########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/d8680a89f986492dbc27c36af3294034db26458f/llms/mixtral/mixtral.py#L163
########
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = llama.Attention(args=args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = llama.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = llama.RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(self,
                 x: mx.array,
                 mask: mx.array = None,
                 cache = None,
                 training_loss: bool = False
                 ) -> mx.array:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r, router_loss = self.feed_forward(self.ffn_norm(h), training_loss)
        out = h + r

        return out, cache, router_loss
    
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

        return self.output(self.norm(h)), cache

    def loss(self,
        inputs: mx.array,
        targets: mx.array,
        loss_masks: mx.array):
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
            h, _, router_loss = layer(h, mask, None, training_loss=True)
            aux_loss += router_loss

        logits = self.output(self.norm(h))
        logits = logits.astype(mx.float32)

        # Calculate the loss
        cross_entropy_loss = nn.losses.cross_entropy(logits, targets) * loss_masks
        num_tokens = loss_masks.sum()
        loss_value = cross_entropy_loss.sum() / num_tokens

        overall_loss = loss_value + aux_loss * self.router_aux_loss_coef

        return overall_loss, None, num_tokens