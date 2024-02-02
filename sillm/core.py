import logging
import pathlib

import mlx.core as mx

import sillm
from sillm.llm import LLM
from sillm.mapping import map_key, map_config

def load(model_path) -> LLM:
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

def load_model_file(model_path) -> LLM:
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
    
def load_gguf_file(model_path) -> LLM:
    """
    Load model from GGUF file.
    Args:
        model_path: Path to GGUF file.
    Returns:
        SiLLM model.
    """
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

def load_model_dir(model_path) -> LLM:
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
        weights = load_mlx_weights(weights_files_npz)
    elif len(weights_files_safetensors) > 0:
        weights = load_mlx_weights(weights_files_safetensors)
    elif len(weights_files_consolidated) > 0:
        weights = load_torch_weights(weights_files_consolidated)
    elif len(weights_files_bin) > 0:
       weights = load_torch_weights(weights_files_bin)
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
    for weights_path in weights_files:
        logging.debug(f"Loading model weights file {weights_path}")

        for k, v in mx.load(str(weights_path)).items():
            k = map_key(k)

            if k is None:
                logging.warning(f"Unknown key: {k}")
            else:
                weights[k] = v

    return weights

def load_torch_weights(weights_files) -> dict:
    """
    Load model weights using PyTorch.
    Args:
        weights_files: List of weights files.
    Returns:
        Model weights.
    """
    weights = {}
    for weights_path in weights_files:
        logging.debug(f"Loading model weights file {weights_path}")

        for k, v in load_torch_file(str(weights_path)).items():
            k = map_key(k)

            if k is None:
                logging.warning(f"Unknown key: {k}")
            else:
                weights[k] = v

    return weights

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
            v = v.to(dtype=torch.float32).numpy()
        else:
            v = v.numpy()

        dtype = getattr(mx,str(v.dtype).split(".")[-1])
        weights[k] = mx.array(v, dtype)

    return weights