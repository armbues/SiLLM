import pathlib
import json
import logging
import dataclasses

import sillm.utils as utils

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
    norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_traditional: bool
    hidden_dim: int = 0
    max_position_embeddings: int = 0
    bos_token_id: int = None
    eos_token_id: int = None
    quantization: dict = None

    def __repr__(self):
        return json.dumps(dataclasses.asdict(self), indent=4)

    @staticmethod
    def load(config_path):
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

        config = utils.map_config(config)

        ArgsClass = ModelArgs
        if "model_type" in config:
            if config["model_type"] in ("llama", "mistral"):
                ArgsClass = LlamaArgs
            elif config["model_type"] == "mixtral":
                ArgsClass = MixtralArgs
            else:
                logging.warn(f"Unknown model type {config['model_type']} - falling back to default config")

        fields = ModelArgs.__annotations__
        if ArgsClass is not ModelArgs:
            fields.update(ArgsClass.__annotations__)

        config = {k:v for k, v in config.items() if k in fields}
        print(config)
        args = ArgsClass(**config)

        logging.info(f"Loaded model config from {config_path}")
        for k, v in dataclasses.asdict(args).items():
            logging.debug(f"Config {k}: {v}")

        return args
    
@dataclasses.dataclass
class LlamaArgs(ModelArgs):
    """
    Llama model arguments.
    """
    rope_theta: float = 10000.0
    rope_traditional: bool = True
    rope_scaling: dict = None

@dataclasses.dataclass
class MixtralArgs(ModelArgs):
    """
    Mixtral model arguments.
    """
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    router_aux_loss_coef: float = 0.001
    moe: dict = None

    def __post_init__(self):
        if self.moe is None:
            self.moe = {
                "num_experts": 8,
                "num_experts_per_token": 2
            }