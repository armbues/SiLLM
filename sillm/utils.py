import re

import mlx.core as mx

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

def map_key(k):
    """
    Map key to MLX naming scheme.
    Args:
        k: Key to map.
    """
    if k in ["tok_embeddings.weight", "norm.weight", "output.weight", "rope.freqs"]:
        return k
    elif k.startswith("layers."):
        return k
    elif k.startswith("model.embed_tokens."):
        return re.sub(r"^model\.embed_tokens\.", "tok_embeddings.", k)
    elif k.startswith("model.norm."):
        return re.sub(r"^model\.norm\.", "norm.", k)
    elif k.startswith("lm_head."):
        return re.sub(r"^lm_head\.", "output.", k)
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        
        k = re.sub(r"^model\.layers", "layers", k)

        if k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"

        k = re.sub(r"\.self_attn\.(q|k|v|o)_proj\.", r".attention.w\1.", k)
        k = re.sub(r"\.mlp\.gate_proj\.", ".feed_forward.w1.", k)
        k = re.sub(r"\.mlp\.down_proj\.", ".feed_forward.w2.", k)
        k = re.sub(r"\.mlp\.up_proj\.", ".feed_forward.w3.", k)

        return k

    return None

def map_config(config):
    """
    Map config to MLX naming scheme.
    Args:
        config: Configuration to map.
    """
    result = {}

    mlx_keys = [
        "model_type",
        "dim",
        "n_layers",
        "n_heads",
        "head_dim",
        "hidden_dim",
        "n_kv_heads",
        "norm_eps",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "bos_token_id",
        "eos_token_id"
    ]
    for key in mlx_keys:
        if key in config:
            result[key] = config[key]

    if "hidden_size" in config:
        result["dim"] = config["hidden_size"]
    if "num_hidden_layers" in config:
        result["n_layers"] = config["num_hidden_layers"]
    if "num_attention_heads" in config:
        result["n_heads"] = config["num_attention_heads"]
    if "intermediate_size" in config:
        result["hidden_dim"] = config["intermediate_size"]
    if "num_key_value_heads" in config:
        result["n_kv_heads"] = config["num_key_value_heads"]
    elif "n_heads" in config:
        result["n_kv_heads"] = config["n_heads"]
    if "rms_norm_eps" in config:
        result["norm_eps"] = config["rms_norm_eps"]

    if result["vocab_size"] <= 0:
        del(result["vocab_size"])
    if "dim" in result and "n_heads" in result:
        result["head_dim"] = result["dim"] // result["n_heads"]

    return result