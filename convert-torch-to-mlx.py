import sys
import logging
import argparse
import glob
import pathlib
import shutil
import json

import tqdm

import torch
import mlx.core as mx

from sillm.utils import map_key, map_config

def load_weights_torch(files_torch):
    weights = {}

    for fpath_weights in files_torch:
        logging.info(f"Loading weights from {fpath_weights}")

        w = torch.load(fpath_weights, map_location="cpu")
        weights.update(w)

    return weights

def load_config(config_path):
    with open(config_path) as f:
        return json.loads(f.read())

def save_weights(weights_path, shard, format="safetensors"):
    if format == "npz":
        mx.savez(weights_path, **shard)
    elif format == "safetensors":
        mx.save_safetensors(weights_path, shard)
    else:
        raise ValueError(f"Unknown format: {format}")
    
def torch_to_mlx(a: torch.Tensor, *, dtype: str = None) -> mx.array:
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
    parser.add_argument("-f", "--output_format", type=str, default="safetensors", help="Output format for weights files (safetensors/npz)")
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
    files_torch_pth = list(input_path.glob("consolidated.*.pth"))
    files_torch_bin = list(input_path.glob("pytorch_model-*.bin"))

    if len(files_torch_pth) > 0:
        weights = load_weights_torch(files_torch_pth)
    elif len(files_torch_bin) > 0:
        weights = load_weights_torch(files_torch_bin)
    else:
        logging.error(f"No weights found in {args.model_input}")
        exit(0)

    # Convert weights to MLX
    max_shard_size = args.max_shard_size * 10**9
    total_params = 0

    shard, shard_size, num_shards = {}, 0, 0
    pbar = tqdm.tqdm(weights.items(), total=len(weights), desc="Converting weights")
    for k, v in pbar:
        k = map_key(k)
        
        total_params += v.numel()
        estimated_size = v.numel() * v.dtype.itemsize

        # Write shard if it exceeds the max shard size
        if shard_size + estimated_size > max_shard_size:
            weights_path = str(output_path / f"weights.{num_shards}.{args.output_format}")
            logging.info(f"Saving shard {weights_path} with {shard_size/10**9:.2f}GB")
            save_weights(weights_path, shard, format=args.output_format)
            shard, shard_size, num_shards = {}, 0, num_shards + 1

        shard[k] = torch_to_mlx(v, dtype=args.dtype)
        shard_size += estimated_size

        # Add missing config parameters
        if k == "output.weight":
            if "vocab_size" not in config:
                config["vocab_size"] = v.shape[-1]
        elif k == "layers.0.feed_forward.w1.weight":
            if "hidden_dim" not in config:
                config["hidden_dim"] = v.shape[0]

    if shard_size > 0:
        if num_shards == 0:
            # Weights fit into one file
            weights_path = str(output_path / f"weights.{args.output_format}")
            logging.info(f"Saving {weights_path} with {shard_size/10**9:.2f}GB")
            save_weights(weights_path, shard, format=args.output_format)
        else:
            # Write remaining shard
            weights_path = str(output_path / f"weights.{num_shards}.{args.output_format}")
            logging.info(f"Saving shard {weights_path} with {shard_size/10**9:.2f}GB")
            save_weights(weights_path, shard, format=args.output_format)

    # Calculate and print total number of parameters
    logging.debug(f"Total parameters: {total_params/10**9:.2f}B")

    # Copy the tokenizer
    tokenizer_sentencepiece = input_path / "tokenizer.model"
    tokenizer_transformers = input_path / "tokenizer.json"
    if tokenizer_sentencepiece.exists():
        logging.info(f"Copying tokenizer: {tokenizer_sentencepiece}")
        shutil.copyfile(str(tokenizer_sentencepiece), str(output_path / "tokenizer.model"))
    elif tokenizer_transformers.exists():
        logging.info(f"Copying tokenizer: {tokenizer_transformers}")
        shutil.copyfile(str(tokenizer_transformers), str(output_path / "tokenizer.json"))
    else:
        logging.error(f"No tokenizer found in {args.model_input}")
        exit(0)

    # Write config file
    for k, v in config.items():
        logging.debug(f"Config {k}: {v}")

    params_path = output_path / "config.json"
    with open(str(params_path), "w") as f:
        f.write(json.dumps(config, indent=4))