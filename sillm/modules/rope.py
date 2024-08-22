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

    if rope_type in ("default", "linear"):
        return nn.RoPE(args.head_dim,
                       traditional=args.rope_traditional,
                       base=args.rope_theta,
                       scale=rope_scale)
    elif rope_type == "llama3":
        return Llama3RoPE(args.head_dim,
                          max_position_embeddings=args.max_position_embeddings,
                          traditional=args.rope_traditional,
                          base=args.rope_theta,
                          scale=rope_scale,
                          rope_scaling=args.rope_scaling)
    elif rope_type in ("su", "longrope"):
        return SuScaledRotaryEmbedding(args.head_dim,
                                       max_position_embeddings=args.max_position_embeddings,
                                       original_max_position_embeddings=args.original_max_position_embeddings,
                                       base=args.rope_theta,
                                       short_factor=args.rope_scaling["short_factor"],
                                       long_factor=args.rope_scaling["long_factor"])
    elif args.rope_scaling["type"] == "yarn":
        raise NotImplementedError("Yarn RoPE is not implemented")
    else:
        raise NotImplementedError(f"Unknown scaling type {rope_scale}")

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
# https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/38143357bf52ce57009ecbd58cf9f0b0029cb393/modeling_phi3.py#L142
# https://github.com/ml-explore/mlx-examples/blob/85dc76f6e0f2cf3ee3d84c211868a6856e163f3f/llms/mlx_lm/models/su_rope.py#L7
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
                 scale: float = 1.0,
                 short_factor: Union[List[float], float] = 1.0,
                 long_factor: Union[List[float], float] = 1.0,
                 ):
        super().__init__()

        self.original_max_position_embeddings = original_max_position_embeddings

        self._inv_freq_short = 1.0 / (
            mx.array(short_factor, dtype=mx.float32)
            * base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )
        self._inv_freq_long = 1.0 / (
            scale
            * mx.array(long_factor, dtype=mx.float32)
            * base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )
        self.scaling_factor = math.sqrt(
            1
            + math.log(max_position_embeddings / self.original_max_position_embeddings)
            / math.log(self.original_max_position_embeddings)
        )

    def _get_cos_sin(self, offset, L):
        position_ids = mx.arange(offset, offset + L, dtype=mx.float32)
        inv_freq = (
            self._inv_freq_long
            if (offset + L) > self.original_max_position_embeddings
            else self._inv_freq_short
        )
        freqs = position_ids[:, None] * inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.scaling_factor
        sin = mx.sin(emb) * self.scaling_factor
        return cos, sin

    def __call__(self, x, offset: int = 0):
        def _rotate_half(_x):
            midpoint = _x.shape[-1] // 2
            x1, x2 = _x[..., :midpoint], _x[..., midpoint:]
            return mx.concatenate([-x2, x1], axis=-1)

        cos, sin = self._get_cos_sin(offset, x.shape[2])
        return (x * cos) + (_rotate_half(x) * sin)