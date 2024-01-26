import logging
import pathlib

import mlx.core as mx

import sillm
import sillm.utils as utils

def load(model_path):
    """
    Load model from directory.
    """
    model_path = pathlib.Path(model_path)

    # Load weights
    weights_files_npz = sorted(list(model_path.glob("weights*.npz")))
    weights_files_safetensors = sorted(list(model_path.glob("*.safetensors")))
    weights_files_consolidated = sorted(list(model_path.glob("consolidated.*.pth")))
    weights_files_bin = sorted(list(model_path.glob("pytorch_model-*.bin")))

    if len(weights_files_npz) > 0:
        weights = load_weights_mlx(weights_files_npz)
    elif len(weights_files_safetensors) > 0:
        weights = load_weights_mlx(weights_files_safetensors)
    elif len(weights_files_consolidated) > 0:
        weights = load_weights_torch(weights_files_consolidated)
    elif len(weights_files_bin) > 0:
       weights = load_weights_torch(weights_files_bin)
    else:
        raise ValueError("No weights files found")
    
    total_params = sum(v.size for v in weights.values())
    logging.info(f"Loaded model weights with {total_params/10**9:.2f}B total parameters")

    # Load configuration
    model_args = None
    for config_file in ("config.json", "params.json"):
        config_path = model_path / config_file
        if config_path.exists():
            model_args = sillm.ModelArgs.load(config_path)
            break
        else:
            logging.debug(f"No config file {config_path} not found")
    if model_args is None:
        raise ValueError(f"Configuration could not be loaded from {model_path}")
    logging.info(f"Loaded model configuration from {config_path}")

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
    model = sillm.LLM(tokenizer, model_args)

    # Verify that all model weights are present
    model.verify_weights(weights)

    # Update model weights
    model.update_weights(weights)

    return model, tokenizer

def load_weights_mlx(weights_files):
    """
    Load model weights using MLX.
    Args:
        weights_files: List of weights files.
    """
    weights = {}
    for weights_path in weights_files:
        logging.debug(f"Loading model weights file {weights_path}")

        for k, v in mx.load(str(weights_path)).items():
            k = utils.map_key(k)

            if k:
                weights[k] = v
            else:
                logging.warning(f"Unknown key: {k}")

    return weights

def load_weights_torch(weights_files):
    """
    Load model weights using PyTorch.
    Args:
        weights_files: List of weights files.
    """
    weights = {}
    for weights_path in weights_files:
        logging.debug(f"Loading model weights file {weights_path}")

        for k, v in load_torch(str(weights_path)).items():
            k = utils.map_key(k)

            if k:
                weights[k] = v
            else:
                logging.warning(f"Unknown key: {k}")

    return weights

def load_torch(weights_path):
    """
    Load PyTorch weights and convert to MLX.
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