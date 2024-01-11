import pathlib
import json
import dataclasses

import mlx.core as mx
import mlx.nn as nn

@dataclasses.dataclass
class ModelArgs:
    model_type: str
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float = 10000.0
    rope_traditional: bool = True
    rope_scaling: dict = None
    moe: dict = None

    @staticmethod
    def load(config_path):
        assert pathlib.Path(config_path).exists(), config_path

        with open(config_path, "r") as f:
            config = json.loads(f.read())
        config = {k:v for k, v in config.items() if k in ModelArgs.__annotations__}

        return ModelArgs(**config)

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
    
    def __call__(self, inputs: mx.array, cache=None):
        raise NotImplementedError(f"Class model.Model is used for inheritance only")