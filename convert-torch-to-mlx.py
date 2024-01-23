import sys
import logging
import argparse
import glob
import pathlib
import shutil
import json

import numpy as np
import tqdm

import torch
import mlx.core as mx

def load_weights_torch(files_torch):
    weights = {}

    for fpath_weights in files_torch:
        logging.info(f"Loading weights from {fpath_weights}")

        w = torch.load(fpath_weights, map_location="cpu")
        weights.update(w)

    return weights
    
def map_keys(keys):
    mapping = {}

    for k1 in keys:
        if k1 in ["tok_embeddings.weight", "norm.weight", "output.weight", "rope.freqs"]:
            k2 = k1
        elif k1 == "model.embed_tokens.weight":
            k2 = "tok_embeddings.weight"
        elif k1 == "model.norm.weight":
            k2 = "norm.weight"
        elif k1 == "lm_head.weight":
            k2 = "output.weight"
        elif k1.startswith("model.layers."):
            layer = k1.split(".")[2]
            
            if k1.endswith(".self_attn.q_proj.weight"):
                k2 = f"layers.{layer}.attention.wq.weight"
            elif k1.endswith(".self_attn.k_proj.weight"):
                k2 = f"layers.{layer}.attention.wk.weight"
            elif k1.endswith(".self_attn.v_proj.weight"):
                k2 = f"layers.{layer}.attention.wv.weight"
            elif k1.endswith(".self_attn.o_proj.weight"):
                k2 = f"layers.{layer}.attention.wo.weight"
            elif k1.endswith(".mlp.gate_proj.weight"):
                k2 = f"layers.{layer}.feed_forward.w1.weight"
            elif k1.endswith(".mlp.down_proj.weight"):
                k2 = f"layers.{layer}.feed_forward.w2.weight"
            elif k1.endswith(".mlp.up_proj.weight"):
                k2 = f"layers.{layer}.feed_forward.w3.weight"
            elif k1.endswith(".input_layernorm.weight"):
                k2 = f"layers.{layer}.attention_norm.weight"
            elif k1.endswith(".post_attention_layernorm.weight"):
                k2 = f"layers.{layer}.ffn_norm.weight"
            elif k1.endswith(".self_attn.rotary_emb.inv_freq"):
                continue
            else:
                logging.warning(f"Unknown key: {k1}")
        elif k1.startswith("layers."):
            k2 = k1
        else:
            logging.warning(f"Unknown key: {k1}")
        
        logging.debug(f"Mapping: {k1} => {k2}")

        mapping[k1] = k2
    
    return mapping

def load_config(config_path):
    with open(config_path) as f:
        return json.loads(f.read())

def map_config(config):
    params = {}

    if "dim" in config:
        params["dim"] = config["dim"]
    elif "hidden_size" in config:
        params["dim"] = config["hidden_size"]

    if "n_layers" in config:
        params["n_layers"] = config["n_layers"]
    elif "num_hidden_layers" in config:
        params["n_layers"] = config["num_hidden_layers"]

    if "n_heads" in config:
        params["n_heads"] = config["n_heads"]
    elif "num_attention_heads" in config:
        params["n_heads"] = config["num_attention_heads"]

    if "head_dim" in config:
        params["head_dim"] = config["head_dim"]
    elif "dim" in params and "n_heads" in params:
        params["head_dim"] = params["dim"] // params["n_heads"]

    if "hidden_dim" in config:
        params["hidden_dim"] = config["hidden_dim"]
    elif "intermediate_size" in params:
        params["hidden_dim"] = config["intermediate_size"]
    
    if "n_kv_heads" in config:
        params["n_kv_heads"] = config["n_kv_heads"]
    elif "num_key_value_heads" in config:
            params["n_kv_heads"] = config["num_key_value_heads"]
    elif "n_heads" in config:
        params["n_kv_heads"] = config["n_heads"]

    if "norm_eps" in config:
        params["norm_eps"] = config["norm_eps"]
    elif "rms_norm_eps" in config:
        params["norm_eps"] = config["rms_norm_eps"]

    if "vocab_size" in config and config.get("vocab_size", -1) > 0:
        params["vocab_size"] = config["vocab_size"]

    if "rope_theta" in config:
        params["rope_theta"] = config["rope_theta"]

    if "rope_scaling" in config:
        params["rope_scaling"] = config["rope_scaling"]

    return params

def torch_to_mx(a: torch.Tensor, *, dtype: str = None) -> mx.array:
    # Map dtype
    if dtype is None:
        dtype = str(v.dtype).split(".")[-1]
    dtype = getattr(mx, dtype)

    # Convert to numpy
    if a.dtype == torch.bfloat16:
        a = a.to(dtype=torch.float32).numpy()
    else:
        a = a.numpy()

    return mx.array(a, dtype)

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Convert LLM models to MLX.")
    parser.add_argument("model_input", type=str, help="The input model directory")
    parser.add_argument("model_output", type=str, help="The output directory to store MLX files")
    parser.add_argument("-t", "--model_type", type=str, default="llama", help="Model type")
    parser.add_argument("-d", "--dtype", type=str, default=None, help="Torch data type to save the model weights")
    parser.add_argument("-s", "--max_shard_size", type=int, default=10, help="Max shard size for weights files in GB")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()
    input_path = pathlib.Path(args.model_input)
    output_path = pathlib.Path(args.model_output)

    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(level=log_level, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

    # Create output directory if it doesn't exist
    if not output_path.exists():
        logging.debug(f"Creating output directory {output_path}")
        output_path.mkdir(parents=True)

    # Load and map config file
    logging.info(f"Loading config from {args.model_input}")
    if (input_path / "config.json").exists():
        config = load_config(str(input_path / "config.json"))
    elif (input_path / "params.json").exists():
        config = load_config(str(input_path / "params.json"))
    else:
        logging.error(f"No configuration file found in {args.model_input}")
        exit(0)
    config = map_config(config)
    config["model_type"] = args.model_type

    # Load weights from input model
    files_torch_pth = sorted(glob.glob(str(input_path / "consolidated.*.pth")))
    files_torch_bin = sorted(glob.glob(str(input_path / "pytorch_model-*.bin")))

    if len(files_torch_pth) > 0:
        weights = load_weights_torch(files_torch_pth)
    elif len(files_torch_bin) > 0:
        weights = load_weights_torch(files_torch_bin)
    else:
        logging.error(f"No weights found in {args.model_input}")
        exit(0)

    # Map keys
    mapping = map_keys(weights.keys())

    # Convert weights to MLX
    max_shard_size = args.max_shard_size * 10**9
    total_params = 0

    shard, shard_size, num_shards = {}, 0, 0
    pbar = tqdm.tqdm(weights.items(), total=len(weights), desc="Converting weights")
    for k, v in pbar:
        if k in mapping:
            total_params += v.numel()
            estimated_size = v.numel() * v.dtype.itemsize

            # Write shard if it exceeds the max shard size
            if shard_size + estimated_size > max_shard_size:
                logging.info(f"Saving shard weights.{num_shards}.npz with {shard_size/10**9:.2f}GB")
                mx.savez(str(output_path / f"weights.{num_shards}.npz"), **shard)
                shard, shard_size, num_shards = {}, 0, num_shards + 1

            shard[mapping[k]] = torch_to_mx(v, dtype=args.dtype)
            shard_size += estimated_size

            # Add missing config parameters
            if mapping[k] == "output.weight":
                if "vocab_size" not in config:
                    config["vocab_size"] = v.shape[-1]
            elif mapping[k] == "layers.0.feed_forward.w1.weight":
                if "hidden_dim" not in config:
                    config["hidden_dim"] = v.shape[0]
        else:
            logging.warning(f"Unmapped key {k}")

    # Write remaining shard
    if shard_size > 0:
        if num_shards == 0:
            logging.info(f"Saving weights.npz with {shard_size/10**9:.2f}GB")
            mx.savez(str(output_path / "weights.npz"), **shard)
        else:
            logging.info(f"Saving shard weights.{num_shards}.npz with {shard_size/10**9:.2f}GB")
            mx.savez(str(output_path / f"weights.{num_shards}.npz"), **shard)

    # Calculate and print total number of parameters
    logging.debug(f"Total parameters: {total_params/10**9:.2f}B")

    # Copy the tokenizer
    tokenizer_path = input_path / "tokenizer.model"
    if tokenizer_path.exists():
        logging.info(f"Copying tokenizer: {tokenizer_path}")
        shutil.copyfile(str(tokenizer_path), str(output_path / "tokenizer.model"))
    else:
        logging.error(f"No tokenizer found in {args.model_input}")
        exit(0)

    # Write config file
    for k, v in config.items():
        logging.debug(f"Config {k}: {v}")

    params_path = output_path / "config.json"
    with open(str(params_path), "w") as f:
        f.write(json.dumps(config, indent=4))