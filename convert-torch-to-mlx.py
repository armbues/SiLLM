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
        print(f"Loading {fpath_weights}")

        w = torch.load(fpath_weights, map_location="cpu")
        weights.update(w)

    return weights
    
def map_keys(keys, verbose=False):
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
                print(f"Unknown key: {k1}")
                raise NotImplementedError
        elif k1.startswith("layers."):
            k2 = k1
        else:
            print(f"Unknown key: {k1}")
            raise NotImplementedError
        
        if verbose:
            print(f"{k1} => {k2}")

        mapping[k1] = k2
    
    return mapping

def load_config(config_path):
    with open(config_path) as f:
        return json.loads(f.read())

def map_config(config, weights):
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
    else:
        params["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
    
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
    else:
        params["vocab_size"] = weights["output.weight"].shape[-1]

    if "rope_theta" in config:
        params["rope_theta"] = config["rope_theta"]

    return params

def torch_to_mx(a: torch.Tensor, *, dtype: str = "float16") -> mx.array:
    if dtype == "bfloat16":
        a = a.to(torch.float32)
    else:
        a = a.to(getattr(torch, dtype))

    return mx.array(a.numpy(), getattr(mx, dtype))

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Convert LLM models to MLX.")
    parser.add_argument("model_input", type=str, help="The input model directory")
    parser.add_argument("model_output", type=str, help="The output directory to store MLX files")
    parser.add_argument("-t", "--model_type", type=str, default="llama", help="Model type")
    parser.add_argument("--dtype", type=str, default="float16", help="Torch data type to load the model weights")
    args = parser.parse_args()
    input_path = pathlib.Path(args.model_input)
    output_path = pathlib.Path(args.model_output)

    # Load weights from input model
    files_torch_pth = sorted(glob.glob(str(input_path / "consolidated.*.pth")))
    files_torch_bin = sorted(glob.glob(str(input_path / "pytorch_model-*.bin")))

    if len(files_torch_pth) > 0:
        weights = load_weights_torch(files_torch_pth)
    elif len(files_torch_bin) > 0:
        weights = load_weights_torch(files_torch_bin)
    else:
        print(f"No weights found in {args.model_input}")
        exit(0)

    # Convert keys and weights
    state = {}
    mapping = map_keys(weights.keys(), verbose=args.verbose)
    for k, v in tqdm.tqdm(weights.items(), total=len(weights), desc="Converting weights"):
        if k in mapping:
            state[mapping[k]] = torch_to_mx(v, dtype=args.dtype)

    np.savez(str(output_path / "weights.npz"), **state)

    # Calculate and print total number of parameters
    num_params = sum(v.size for v in state.values()) / 10**9
    print(f"Total parameters: {num_params:.2f}B")

    # Copy the tokenizer
    tokenizer_path = input_path / "tokenizer.model"
    if tokenizer_path.exists():
        print(f"Copying tokenizer: {tokenizer_path}")
        shutil.copyfile(str(tokenizer_path), str(output_path / "tokenizer.model"))
    else:
        print(f"Make sure there is a file tokenizer.model in {args.torch_model}")
        exit(0)

    # Load and convert config file
    if (input_path / "config.json").exists():
        config = load_config(str(input_path / "config.json"))
    elif (input_path / "params.json").exists():
        config = load_config(str(input_path / "params.json"))
    else:
        print(f"No configuration file found in {args.torch_model}")
        exit(0)
        
    config = map_config(config, state)
    config["model_type"] = args.model_type
    print(json.dumps(config, indent=4))

    params_path = output_path / "config.json"
    with open(str(params_path), "w") as f:
        f.write(json.dumps(config, indent=4))