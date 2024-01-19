import pathlib
import json
import logging
import dataclasses

@dataclasses.dataclass
class ModelArgs:
    """
    Model arguments.
    """
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
        config = {k:v for k, v in config.items() if k in ModelArgs.__annotations__}

        logging.info(f"Loaded model config from {config_path}")
        for k, v in dataclasses.asdict(args).items():
            logging.debug(f"Config {k}: {v}")

        if "model_type" in config:
            if config["model_type"] == "mistral":
                args = MixtralArgs(**config)
        
        return ModelArgs(**config)
    
class MixtralArgs(ModelArgs):
    """
    Mixtral model arguments.
    """
    router_aux_loss_coef: float = 0.001
    rope_theta: float = 1000000.0