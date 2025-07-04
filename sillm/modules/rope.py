import math
from typing import Union, List

import mlx.core as mx
import mlx.nn as nn

from sillm.models.args import ModelArgs

def init_rope(args: ModelArgs):
    rope_type = "default"
    rope_scale = 1.0
    if args.rope_scaling is not None:
        rope_type = (args.rope_scaling.get("type") or args.rope_scaling.get("rope_type") or "default")

        if rope_type == "linear":
            rope_scale = 1 / args.rope_scaling["factor"]

    rope_dims = args.head_dim
    if args.partial_rotary_factor is not None:
        rope_dims = int(rope_dims * args.partial_rotary_factor)

    if rope_type in ("default", "linear"):
        return nn.RoPE(rope_dims,
                       traditional=args.rope_traditional,
                       base=args.rope_theta,
                       scale=rope_scale)
    elif rope_type == "llama3":
        return Llama3RoPE(rope_dims,
                          max_position_embeddings=args.max_position_embeddings,
                          traditional=args.rope_traditional,
                          base=args.rope_theta,
                          scale=rope_scale,
                          rope_scaling=args.rope_scaling)
    elif rope_type in ("su", "longrope"):
        return SuScaledRotaryEmbedding(rope_dims,
                                       max_position_embeddings=args.max_position_embeddings,
                                       original_max_position_embeddings=args.original_max_position_embeddings,
                                       base=args.rope_theta,
                                       rope_scaling=args.rope_scaling)
    elif rope_type == "yarn":
        raise NotImplementedError("Yarn RoPE is not implemented yet.")
    else:
        raise NotImplementedError(f"Unknown scaling type {rope_type}")

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/85dc76f6e0f2cf3ee3d84c211868a6856e163f3f/llms/mlx_lm/models/llama.py#L49
########
class Llama3RoPE(nn.Module):
    def __init__(self,
                 head_dim: int,
                 max_position_embeddings: int = 131072,
                 traditional: bool = True,
                 base: float = 10000,
                 scale: float = 1.0,
                 rope_scaling: dict = None,
                 ):
        super().__init__()

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional
        self.original_base = base
        self.scale = scale

        ########
        # Calculate base frequencies
        # References:
        # https://github.com/huggingface/transformers/blob/d5a99dfcee6e94065cb7c83cc8ab6fc5daa0cc4e/src/transformers/modeling_rope_utils.py#L318
        ########
        factor = rope_scaling.get("factor", 8.0)
        low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
        old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = self.original_base ** (mx.arange(0, head_dim, 2) / head_dim)
        wavelens = 2 * mx.pi * freqs
        new_base_freqs = []

        smooths = (wavelens - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
        new_base_freqs = freqs * (1 - smooths) * factor + smooths
        new_base_freqs = mx.where(wavelens < high_freq_wavelen, freqs, new_base_freqs)
        new_base_freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, new_base_freqs)

        self.base = new_base_freqs.mean().item()
            
    def __call__(self, x, offset: int = 0):
        seq_len = x.shape[1] + offset
        base = self.base
        if self.max_position_embeddings and seq_len > self.max_position_embeddings:
            base *= (
                (self.scale * seq_len / self.max_position_embeddings) - (self.scale - 1)
            ) ** (self.head_dim / (self.head_dim - 2))

        return mx.fast.rope(
            x,
            self.head_dim,
            traditional=self.traditional,
            base=base,
            scale=self.scale,
            offset=offset,
        )

########
# Su Scaled Rotary Embedding
# References:
# https://github.com/ml-explore/mlx-examples/blob/00a73790702991075b80f9facf219ae397e1eb15/llms/mlx_lm/models/su_rope.py#L10
########
class SuScaledRotaryEmbedding(nn.Module):
    """
    Scaled Uniform RoPE.
    """
    def __init__(self,
                 head_dim: int,
                 max_position_embeddings: int = 131072,
                 original_max_position_embeddings: int = 4096,
                 base: float = 10000,
                 rope_scaling: dict = None,
                 ):
        super().__init__()

        short_factor = rope_scaling.get("short_factor", 1.0)
        long_factor = rope_scaling.get("long_factor", 1.0)
        short_mscale = rope_scaling.get("short_mscale", None)
        long_mscale = rope_scaling.get("long_mscale", None)

        self.head_dim = head_dim
        
        freqs = base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        self._freqs = mx.array(long_factor, dtype=mx.float32) * freqs
        self.scale = long_mscale or math.sqrt(1 + math.log(max_position_embeddings / original_max_position_embeddings) / math.log(self.original_max_position_embeddings))

    def __call__(self, x, offset: int = 0):
        x[..., : self.head_dim] = self.scale * x[..., : self.head_dim]

        return mx.fast.rope(
            x,
            self.head_dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs
        )