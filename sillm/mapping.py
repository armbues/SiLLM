import re

import mlx.core as mx

def map_key(k):
    """
    Map key to MLX naming scheme.
    Args:
        k: Key to map.
    """
    if k.startswith("layers."):
        return k
    elif k.startswith("output."):
        return k
    elif k in ["tok_embeddings.weight", "norm.weight", "rope.freqs"]:
        return k
    elif k.startswith("model.embed_tokens."):
        return re.sub(r"^model\.embed_tokens\.", "tok_embeddings.", k)
    elif k.startswith("model.norm."):
        return re.sub(r"^model\.norm\.", "norm.", k)
    elif k.startswith("lm_head."):
        return re.sub(r"^lm_head\.", "output.", k)
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        
        k = re.sub(r"^model\.layers\.", "layers.", k)

        if k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"

        k = re.sub(r"\.self_attn\.(q|k|v|o)_proj\.", r".attention.w\1.", k)
        k = re.sub(r"\.mlp\.gate_proj\.", ".feed_forward.w1.", k)
        k = re.sub(r"\.mlp\.down_proj\.", ".feed_forward.w2.", k)
        k = re.sub(r"\.mlp\.up_proj\.", ".feed_forward.w3.", k)

        return k
    # GGUF keys
    elif k.startswith("output_norm."):
        return re.sub(r"^output_norm\.", "norm.", k)
    elif k.startswith("token_embd."):
        return re.sub(r"^token_embd\.", "tok_embeddings.", k)
    elif k.startswith("blk."):
        layer = k.split(".")[1]

        k = re.sub(r"^blk\.", "layers.", k)

        if k.endswith(".attn_norm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".ffn_norm.weight"):
            return f"layers.{layer}.ffn_norm.weight"

        k = re.sub(r"\.attn_(q|k|v)\.", r".attention.w\1.", k)
        k = re.sub(r"\.attn_output\.", r".attention.wo.", k)
        k = re.sub(r"\.ffn_gate\.", ".feed_forward.w1.", k)
        k = re.sub(r"\.ffn_down\.", ".feed_forward.w2.", k)
        k = re.sub(r"\.ffn_up\.", ".feed_forward.w3.", k)

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

    key_map = {
        "hidden_size": "dim",
        "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads",
        "intermediate_size": "hidden_dim",
        "num_key_value_heads": "n_kv_heads",
        "n_heads": "n_kv_heads",
        "rms_norm_eps": "norm_eps",
        # GGUF metadata
        "llama.embedding_length": "dim",
        "llama.block_count": "n_layers",
        "llama.attention.head_count": "n_heads",
        "llama.feed_forward_length": "hidden_dim",
        "llama.attention.head_count_kv": "n_kv_heads",
        "llama.attention.layer_norm_rms_epsilon": "norm_eps",
        "llama.rope.freq_base": "rope_theta"
    }

    for key in key_map:
        if key in config:
            value = config[key]
            if isinstance(value, mx.array):
                value = value.item()
            result[key_map[key]] = value

    if "general.architecture" in config:
        result["model_type"] = config["general.architecture"]
        if "general.name" in config and "mixtral" in config["general.name"].lower():
            result["model_type"] = "mixtral"
    if "tokenizer.ggml.tokens" in config:
        result["vocab_size"] = len(config["tokenizer.ggml.tokens"])
    if result["vocab_size"] <= 0:
        del(result["vocab_size"])
    if "llama.expert_count" in config and "llama.expert_used_count" in config:
        result["moe"] = {
            "num_experts": config["llama.expert_count"].item(),
            "num_experts_per_tok": config["llama.expert_used_count"].item()
        }

    # Calculate head_dim if not given
    if "head_dim" not in result:
        if "dim" in result and "n_heads" in result:
            result["head_dim"] = result["dim"] // result["n_heads"]

    return result