import logging
import pathlib
import enum

import mlx.core as mx

from .llm import LLM
from .tokenizer import GGUFTokenizer, TransformerTokenizer, SentencePieceTokenizer, TiktokenTokenizer
from sillm.models.args import ModelArgs
from sillm.utils.mapping import map_key, map_config

logger = logging.getLogger("sillm")

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
            elif k.startswith("model.layers.") or k.startswith("transformer."):
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
    logger.debug(f"Loading GGUF file {model_path}")
    gguf_weights, metadata = mx.load(model_path, return_metadata=True)

    # Map metadata to configuration
    config = map_config(metadata)
    model_args = ModelArgs.load_config(config)

    # Map weights keys
    weights = {}
    mapping = {}
    for gguf_key, value in gguf_weights.items():
        mlx_key = map_key(gguf_key)
        mapping[mlx_key] = gguf_key

        if mlx_key is None:
            logger.warn(f"Unknown key: {gguf_key}")
        else:
            weights[mlx_key] = value

    # Map quantization configuration
    gguf_file_type = metadata["general.file_type"].item()
    logger.debug(f"GGUF file type: {gguf_file_type}")
    quantization = None
    if gguf_file_type == 0 or gguf_file_type == 1:
        # No quantization
        pass
    elif gguf_file_type == 2 or gguf_file_type == 3:
        quantization = {"group_size": 32, "bits": 4}
    elif gguf_file_type == 7:
        quantization = {"group_size": 32, "bits": 8}
    else:
        logger.warn(f"Unsupported GGUF file type: {gguf_file_type}")
    model_args.quantization = quantization

    # Fix configuration
    model_args.fix_config(weights)
    model_args.log_config()

    # Load tokenizer
    tokenizer = GGUFTokenizer(metadata)
    logger.info("Loaded tokenizer from GGUF metadata")

    # Initialize model
    model = LLM(tokenizer, model_args)
    model.init_description(model_path)

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
    
    # Dequantize token embedding weights
    dequantize("tok_embeddings")

    # Verify that all model weights are present
    model.verify_weights(weights)

    # Update model weights
    model.update_weights(weights, mapping=mapping)

    total_params = sum(v.size for v in weights.values())
    logger.info(f"Loaded model weights with {total_params/10**9:.2f}B total parameters")

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

    # Load configuration
    model_args = None
    for config_file in ("config.json", "params.json"):
        config_path = model_path / config_file
        if config_path.exists():
            model_args = ModelArgs.load_file(config_path)
            break
        else:
            logger.debug(f"No config file {config_path} not found")
    if model_args is None:
        raise ValueError(f"Configuration could not be loaded from {model_path}")
    logger.info(f"Loaded model config from {config_path}")

    # Load weights
    weights_files_npz = sorted(list(model_path.glob("weights*.npz")))
    weights_files_safetensors = sorted(list(model_path.glob("*.safetensors")))
    weights_files_consolidated = sorted(list(model_path.glob("consolidated.*.pth")))
    weights_files_bin = sorted(list(model_path.glob("pytorch_model-*.bin")))

    if len(weights_files_npz) > 0:
        weights, mapping, model_format = load_weights(weights_files_npz, load_func=mx.load)
    elif len(weights_files_safetensors) > 0:
        weights, mapping, model_format = load_weights(weights_files_safetensors, load_func=mx.load)
    elif len(weights_files_consolidated) > 0:
        weights, mapping, model_format = load_weights(weights_files_consolidated, load_func=load_torch_file)
    elif len(weights_files_bin) > 0:
       weights, mapping, model_format = load_weights(weights_files_bin, load_func=load_torch_file)
    else:
        raise ValueError("No weights files found")

    if model_format == ModelFormat.HUGGINGFACE:
        logger.debug("Disabling rope_traditional for HuggingFace model")

        model_args.rope_traditional = False

    # Fix configuration
    model_args.fix_config(weights)
    model_args.log_config()

    # Load tokenizer
    tokenizer = None
    tokenizer_path = None
    if (model_path / "tokenizer.model").exists():
        tokenizer_path = model_path / "tokenizer.model"
        tokenizer = SentencePieceTokenizer(str(tokenizer_path), model_args)
    elif (model_path / "tokenizer.json").exists():
        tokenizer_path = model_path / "tokenizer.json"
        tokenizer = TransformerTokenizer(str(model_path), model_args)
    elif model_args.model_type == "dbrx":
        tokenizer_path = "tiktoken"
        tokenizer = TiktokenTokenizer(model_args)
    if tokenizer is None:
        logger.error(f"No tokenizer found in {model_path}")
    logger.info(f"Loaded tokenizer from {tokenizer_path}")

    # Initialize model
    model = LLM(tokenizer, model_args)
    model.init_description(model_path)

    # Quantize model
    if model_args.quantization is not None:
        model.quantize(group_size=model_args.quantization["group_size"], bits=model_args.quantization["bits"])

    # Verify that all model weights are present
    model.verify_weights(weights)

    # Update model weights
    model.update_weights(weights, mapping=mapping)

    total_params = sum(v.size for v in weights.values())
    logger.info(f"Loaded model weights with {total_params/10**9:.2f}B total parameters")

    return model

def load_weights(weights_files,
                 load_func: callable
                 ):
    """
    Load model weights.
    Args:
        weights_files: List of weights files.
        loader: File loader function.
    Returns:
        Model weights.
    """
    weights = {}
    mapping = {}
    format = ModelFormat.UNKNOWN
    for weights_path in weights_files:
        logger.debug(f"Loading model weights file {weights_path}")
        weights_shard = load_func(str(weights_path))

        # Guess model format according to key names
        if format == ModelFormat.UNKNOWN:
            format = ModelFormat.guess_from_weights(weights_shard)
            logger.debug(f"Guessing model format: {format}")

        for key, value in weights_shard.items():
            mlx_key = map_key(key)
            mapping[mlx_key] = key
            
            mx.eval(value)

            if mlx_key is None:
                logger.warning(f"Unknown key: {key}")
            else:
                if mlx_key in weights:
                    logger.warning(f"Duplicate key: {mlx_key} {value.shape}")

                weights[mlx_key] = value

    return weights, mapping, format

def load_torch_file(weights_path):
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
    for key, value in torch.load(weights_path, map_location="cpu").items():
        # Convert to numpy
        if value.dtype == torch.bfloat16:
            value = value.to(dtype=torch.float16).numpy()
        else:
            v = v.numpy()

        weights[key] = mx.array(v, dtype=mx.float16)

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