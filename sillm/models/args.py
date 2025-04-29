import pathlib
import json
import logging
import dataclasses

from typing import Union

from sillm.utils.mapping import map_config

logger = logging.getLogger("sillm")

@dataclasses.dataclass
class ModelArgs:
    """
    Model arguments.
    """
    model_type: str
    dim: int
    n_layers: int
    head_dim: int
    n_heads: int
    n_kv_heads: int = None
    norm_eps: float = 1e-5
    hidden_dim: int = None
    vocab_size: int = -1
    rope_theta: float = 10000.0
    rope_traditional: bool = True
    rope_scaling: dict = None
    partial_rotary_factor: float = None
    hidden_act: str = None
    max_position_embeddings: int = 0
    original_max_position_embeddings: int = 0
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    bos_token_id: int = None
    eos_token_id: Union[int, list] = None
    pad_token_id: int = None
    quantization: dict = None

    def __repr__(self):
        return json.dumps(dataclasses.asdict(self), indent=4)
    
    def log_config(self):
        for k, v in dataclasses.asdict(self).items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    logger.debug(f"Config {k}.{k2}: {v2}")
            else:
                logger.debug(f"Config {k}: {v}")
    
    def fix_config(self, weights):
        """
        Fix config with shape information from weights.
        """
        if self.hidden_dim is None and "layers.0.feed_forward.w1.weight" in weights:
            self.hidden_dim = weights["layers.0.feed_forward.w1.weight"].shape[0]
        if self.vocab_size <= 0 and "output.weight" in weights:
            self.vocab_size = weights["output.weight"].shape[0]
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

    def save_config(self, config_path):
        """
        Save model config to JSON file.
        Args:
            config_path: Path to config file.
        """
        config = dataclasses.asdict(self)
        
        # Remove None values
        for k in list(config.keys()):
            if config[k] is None:
                del config[k]

        with open(config_path, "w") as f:
            f.write(json.dumps(config, indent=4))

    @staticmethod
    def load_config(config):
        ArgsClass = None

        args_map = {
            "llama": ModelArgs,
            "mistral": ModelArgs,
            "gemma": ModelArgs,
            "mixtral": MixtralArgs,
            "phi": ModelArgs,
            "qwen2": ModelArgs,
            "qwen3": Qwen3Args,
            "starcoder2": ModelArgs,
            "dbrx": DbrxArgs,
            "cohere": CohereArgs,
            "phi3": Phi3Args,
            "gemma2": Gemma2Args,
            "gemma3_text": Gemma3Args,
            "phimoe": PhiMoEArgs,
            "pharia-v1": ModelArgs,
            "granite": GraniteArgs,
            "mobilellm": ModelArgs,
            "exaone": ModelArgs,
        }

        if "model_type" in config:
            model_type = config["model_type"]

            if model_type in args_map:
                ArgsClass = args_map[model_type]
            else:
                logger.warning(f"Unknown model type {model_type} - falling back to `default` config")
                ArgsClass = ModelArgs
        if ArgsClass is None:
            ArgsClass = ModelArgs
            config["model_type"] = "llama"
            logger.warning(f"No model type specified - falling back to `default` config")

        fields = ModelArgs.__annotations__
        fields.update(ArgsClass.__annotations__)
        config = {k:v for k, v in config.items() if k in fields}

        return ArgsClass(**config)
    
    @staticmethod
    def load_file(config_path):
        """
        Load model config from JSON file.
        Args:
            config_path: Path to config file.
        Returns:
            ModelArgs instance.
        """
        assert pathlib.Path(config_path).exists(), config_path

        with open(config_path, "r") as f:
            config = json.loads(f.read())
        if "text_config" in config:
            config = config["text_config"]
        config = map_config(config)

        return ModelArgs.load_config(config)

@dataclasses.dataclass
class MixtralArgs(ModelArgs):
    """
    Mixtral model arguments.
    """
    rope_theta: float = 1000000.0
    router_aux_loss_coef: float = 0.001
    moe: dict = None

    def __post_init__(self):
        if self.moe is None:
            self.moe = {
                "num_experts": 8,
                "num_experts_per_tok": 2
            }

@dataclasses.dataclass
class DbrxArgs(ModelArgs):
    """
    DBRX model arguments.
    """
    clip_qkv: int = 8
    rope_theta: float = 500000.0
    router_aux_loss_coef: float = 0.05
    moe: dict = None

    def __post_init__(self):
        if self.moe is None:
            self.moe = {
                "num_experts": 16,
                "num_experts_per_tok": 4
            }

        if self.bos_token_id is None:
            self.bos_token_id = 100257
        if self.eos_token_id is None:
            self.eos_token_id = 100257

@dataclasses.dataclass
class CohereArgs(ModelArgs):
    """
    Cohere model arguments.
    """
    norm_bias: bool = False
    use_qk_norm: bool = False
    logit_scale: float = 0.0625

@dataclasses.dataclass
class Phi3Args(ModelArgs):
    """
    Phi-3 model arguments.
    """
    embd_pdrop: float = 0.0

@dataclasses.dataclass
class Gemma2Args(ModelArgs):
    """
    Gemma 2 model arguments.
    """
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0
    query_pre_attn_scalar: float = 144.0
    
@dataclasses.dataclass
class Gemma3Args(ModelArgs):
    """
    Gemma 3 model arguments.
    """
    head_dim: int = 256
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 262208
    rms_norm_eps: float = 1e-6
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    rope_local_base_freq: float = 10000.0
    rope_global_base_freq: float = 1000000.0

@dataclasses.dataclass
class PhiMoEArgs(ModelArgs):
    """
    Phi-MoE model arguments.
    """
    num_local_experts: int = 16
    num_experts_per_tok: int = 2
    rms_norm_eps: float = 1e-5

@dataclasses.dataclass
class GraniteArgs(ModelArgs):
    """
    Granite model arguments.
    """
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    attention_multiplier: float = 1.0
    logits_scaling: float = 1.0

class Qwen3Args(ModelArgs):
    """
    Qwen 3 model arguments.
    """
    rms_norm_eps: float = 1e-6