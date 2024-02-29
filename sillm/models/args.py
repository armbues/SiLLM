import pathlib
import json
import logging
import dataclasses

from sillm.mapping import map_config

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
    n_kv_heads: int
    norm_eps: float = None
    hidden_dim: int = None
    vocab_size: int = -1
    rope_theta: float = 10000.0
    rope_traditional: bool = True
    max_position_embeddings: int = 0
    bos_token_id: int = None
    eos_token_id: int = None
    pad_token_id: int = None
    quantization: dict = None

    def __repr__(self):
        return json.dumps(dataclasses.asdict(self), indent=4)
    
    def log_config(self):
        for k, v in dataclasses.asdict(self).items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    logging.debug(f"Config {k}.{k2}: {v2}")
            else:
                logging.debug(f"Config {k}: {v}")
    
    def fix_config(self, weights):
        """
        Fix config with shape information from weights.
        """
        if self.hidden_dim is None and "layers.0.feed_forward.w1.weight" in weights:
            self.hidden_dim = weights["layers.0.feed_forward.w1.weight"].shape[0]
        if self.vocab_size <= 0 and "output.weight" in weights:
            self.vocab_size = weights["output.weight"].shape[0]

    @staticmethod
    def load_config(config):
        ArgsClass = None
        if "model_type" in config:
            if config["model_type"] in ("llama", "mistral", "gemma"):
                ArgsClass = LlamaArgs
            elif config["model_type"] == "mixtral":
                ArgsClass = MixtralArgs
            elif config["model_type"] == "phi":
                ArgsClass = PhiArgs
        if ArgsClass is None:
            config["model_type"] = "llama"
            ArgsClass = LlamaArgs
            logging.warn(f"No model type specified - falling back to default model type `llama`")

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
        config = map_config(config)

        return ModelArgs.load_config(config)
    
@dataclasses.dataclass
class LlamaArgs(ModelArgs):
    """
    Llama model arguments.
    """
    rope_scaling: dict = None

@dataclasses.dataclass
class MixtralArgs(ModelArgs):
    """
    Mixtral model arguments.
    """
    rope_theta: float = 1000000.0
    rope_scaling: dict = None
    router_aux_loss_coef: float = 0.001
    moe: dict = None

    def __post_init__(self):
        if self.moe is None:
            self.moe = {
                "num_experts": 8,
                "num_experts_per_tok": 2
            }

@dataclasses.dataclass
class PhiArgs(ModelArgs):
    """
    Phi model arguments.
    """
    rope_traditional: bool = False
    rope_scaling: dict = None
    partial_rotary_factor: float = 0.4