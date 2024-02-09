import logging
import pathlib
import enum

import mlx.core as mx

import sillm
from sillm.llm import LLM
from sillm.args import ModelArgs
from sillm.mapping import map_key, map_config

class ModelFormat(enum.Enum):
    """
    Model type enumeration.
    """
    UNKNOWN = 0
    MLX = 1
    GGUF = 2
    HUGGINGFACE = 3

    @staticmethod
    def guess_from_weights(weights: dict):
        """
        Guess model type from weights.
        Args:
            weights: Model weights.
        Returns:
            Model type.
        """
        for k in weights:
            if k.startswith("layers."):
                return ModelFormat.MLX
            elif k.startswith("blk."):
                return ModelFormat.GGUF
            elif k.startswith("model.layers."):
                return ModelFormat.HUGGINGFACE
            
        return ModelFormat.UNKNOWN

def load(model_path: str) -> LLM:
    """
    Load model from directory.
    Args:
        model_path: Path to model directory.
    Returns:
        SiLLM model.
    """
    model_path = pathlib.Path(model_path)

    if model_path.is_dir():
        return load_model_dir(str(model_path))
    elif model_path.is_file():
        return load_model_file(str(model_path))
    else:
        raise ValueError(f"Model path {model_path} is not a file or directory")

def load_model_file(model_path: str) -> LLM:
    """
    Load model from file.
    Args:
        model_path: Path to model file.
    """
    model_path = pathlib.Path(model_path)

    if model_path.suffix == ".gguf":
        return load_gguf_file(str(model_path))
    else:
        raise ValueError(f"Unknown model file type: {model_path}")
    
def load_gguf_file(model_path: str) -> LLM:
    """
    Load model from GGUF file.
    Args:
        model_path: Path to GGUF file.
    Returns:
        SiLLM model.
    """
    logging.debug(f"Loading GGUF file {model_path}")
    gguf_weights, metadata = mx.load(model_path, return_metadata=True)

    # Map weights keys
    weights = {}
    for gguf_key, value in gguf_weights.items():
        mlx_key = map_key(gguf_key)

        if mlx_key is None:
            logging.warn(f"Unknown key: {gguf_key}")
        else:
            weights[mlx_key] = value

    # Map metadata and load configuration
    config = map_config(metadata)
    model_args = sillm.ModelArgs.load_config(config)

    # Map quantization configuration
    gguf_file_type = metadata["general.file_type"].item()
    logging.debug(f"GGUF file type: {gguf_file_type}")
    quantization = None
    if gguf_file_type == 0 or gguf_file_type == 1:
        # No quantization
        pass
    elif gguf_file_type == 2 or gguf_file_type == 3:
        quantization = {"group_size": 32, "bits": 4}
    elif gguf_file_type == 7:
        quantization = {"group_size": 32, "bits": 8}
    else:
        logging.warn(f"Unsupported GGUF file type: {gguf_file_type}")
    model_args.quantization = quantization

    # Fix configuration
    model_args.fix_config(weights)
    model_args.log_config()

    # Load tokenizer
    tokenizer = sillm.tokenizer.GGUFTokenizer(metadata)
    logging.info("Loaded tokenizer from GGUF metadata")

    # Initialize model
    model = LLM(tokenizer, model_args)

    # Quantize model
    if quantization is not None:
        excluded = []
        if "output.scales" not in weights:
            excluded = ["output"]
        model.quantize(group_size=model_args.quantization["group_size"], bits=model_args.quantization["bits"], excluded=excluded)

    def dequantize(k):
        weight = weights.pop(f"{k}.weight")
        scales = weights.pop(f"{k}.scales")
        biases = weights.pop(f"{k}.biases")
        weights[f"{k}.weight"] = mx.dequantize(weight, scales=scales, biases=biases, **quantization)
    dequantize("tok_embeddings")

    # Verify that all model weights are present
    model.verify_weights(weights)

    # Update model weights
    model.update_weights(weights)

    total_params = sum(v.size for v in weights.values())
    logging.info(f"Loaded model weights with {total_params/10**9:.2f}B total parameters")

    return model

def load_model_dir(model_path: str) -> LLM:
    """
    Load model from directory.
    Args:
        model_path: Path to model directory.
    Returns:
        SiLLM model.
    """
    model_path = pathlib.Path(model_path)

    # Load weights
    weights_files_npz = sorted(list(model_path.glob("weights*.npz")))
    weights_files_safetensors = sorted(list(model_path.glob("*.safetensors")))
    weights_files_consolidated = sorted(list(model_path.glob("consolidated.*.pth")))
    weights_files_bin = sorted(list(model_path.glob("pytorch_model-*.bin")))

    if len(weights_files_npz) > 0:
        weights, model_format = load_mlx_weights(weights_files_npz)
    elif len(weights_files_safetensors) > 0:
        weights, model_format = load_mlx_weights(weights_files_safetensors)
    elif len(weights_files_consolidated) > 0:
        weights, model_format = load_torch_weights(weights_files_consolidated)
    elif len(weights_files_bin) > 0:
       weights, model_format = load_torch_weights(weights_files_bin)
    else:
        raise ValueError("No weights files found")

    # Load configuration
    model_args = None
    for config_file in ("config.json", "params.json"):
        config_path = model_path / config_file
        if config_path.exists():
            model_args = sillm.ModelArgs.load_file(config_path)
            break
        else:
            logging.debug(f"No config file {config_path} not found")
    if model_args is None:
        raise ValueError(f"Configuration could not be loaded from {model_path}")
    logging.info(f"Loaded model config from {config_path}")

    if model_format == ModelFormat.HUGGINGFACE:
        logging.debug("Permuting HuggingFace weights")

        permute_hf_weights(weights, model_args)

    # Fix configuration
    model_args.fix_config(weights)
    model_args.log_config()

    # Load tokenizer
    tokenizer = None
    tokenizer_path = model_path / "tokenizer.model"
    if tokenizer_path.exists():
        tokenizer = sillm.tokenizer.SentencePieceTokenizer(str(tokenizer_path), model_args)
    else:
        tokenizer_path = model_path / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = sillm.tokenizer.TransformerTokenizer(str(model_path), model_args)
    if tokenizer is None:
        pass
    logging.info(f"Loaded tokenizer from {tokenizer_path}")

    # Initialize model
    model = LLM(tokenizer, model_args)

    # Quantize model
    if model_args.quantization is not None:
        model.quantize(group_size=model_args.quantization["group_size"], bits=model_args.quantization["bits"])

    # Verify that all model weights are present
    model.verify_weights(weights)

    # Update model weights
    model.update_weights(weights)

    total_params = sum(v.size for v in weights.values())
    logging.info(f"Loaded model weights with {total_params/10**9:.2f}B total parameters")

    return model

def load_mlx_weights(weights_files) -> dict:
    """
    Load model weights using MLX.
    Args:
        weights_files: List of weights files.
    Returns:
        Model weights.
    """
    weights = {}
    format = ModelFormat.UNKNOWN
    for weights_path in weights_files:
        logging.debug(f"Loading model weights file {weights_path}")
        weights_shard = mx.load(str(weights_path))

        # Guess model format according to key names
        if format == ModelFormat.UNKNOWN:
            format = ModelFormat.guess_from_weights(weights_shard)
            logging.info(f"Guessing model format: {format}")

        for key, value in weights_shard.items():
            mlx_key = map_key(key)
            mx.eval(value)

            if mlx_key is None:
                logging.warning(f"Unknown key: {key}")
            else:
                if mlx_key in weights:
                    logging.warning(f"Duplicate key: {mlx_key}")

                weights[mlx_key] = value

    return weights, format

def load_torch_weights(weights_files) -> dict:
    """
    Load model weights using PyTorch.
    Args:
        weights_files: List of weights files.
    Returns:
        Model weights.
    """
    weights = {}
    format = ModelFormat.UNKNOWN
    for weights_path in weights_files:
        logging.debug(f"Loading model weights file {weights_path}")
        weights_shard = load_torch_file(str(weights_path))

        # Guess model format according to key names
        if format == ModelFormat.UNKNOWN:
            format = ModelFormat.guess_from_weights(weights_shard)
            logging.info(f"Guessing model format: {format}")

        for key, value in weights_shard.items():
            mlx_key = map_key(key)

            if mlx_key is None:
                logging.warning(f"Unknown key: {key}")
            else:
                if mlx_key in weights:
                    logging.warning(f"Duplicate key: {mlx_key}")

                weights[mlx_key] = value

    return weights, format

def load_torch_file(weights_path) -> dict:
    """
    Load PyTorch weights and convert to MLX.
    Args:
        weights_path: Path to weights file.
    Returns:
        Model weights.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("Please install torch library to load PyTorch weights")
    
    weights = {}
    for k, v in torch.load(weights_path, map_location="cpu").items():
        # Convert to numpy
        if v.dtype == torch.bfloat16:
            v = v.to(dtype=torch.float16).numpy()
        else:
            v = v.numpy()

        weights[k] = mx.array(v, dtype=mx.float16)

    return weights

########
# Based on:
# https://github.com/ggerganov/llama.cpp/blob/b7b74cef36a93ae01e0b9af8986d131761742d0e/convert.py#L1182
########
def permute_hf_weights(weights: dict,
                       args: ModelArgs
                       ):
    """
    Permute weights stored in huggingface format.
    """
    def permute(x, n):
        return (x.reshape(n, 2, x.shape[0] // n // 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape))

    n_heads = args.n_heads
    n_kv_heads = args.n_kv_heads

    for i in range(args.n_layers):
        if f"layers.{i}.attention.wq.weight" in weights:
            weights[f"layers.{i}.attention.wq.weight"] = permute(weights[f"layers.{i}.attention.wq.weight"], n_heads)
        if f"layers.{i}.attention.wk.weight" in weights:
            weights[f"layers.{i}.attention.wk.weight"] = permute(weights[f"layers.{i}.attention.wk.weight"], n_kv_heads)