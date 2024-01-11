import argparse
import pathlib
import shutil
import json

import mlx.core as mx

import torch
import numpy as np
import tqdm

def load_config(config_path):
    with open(config_path) as f:
        print(f"Loading config: {config_path}")
        
        return json.loads(f.read())

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Convert LLM models to MLX.")
    parser.add_argument("model_path", type=str, help="Model directory")
    # parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Increase output verbosity")
    args = parser.parse_args()
    model_path = pathlib.Path(args.model_path)

    # Load MLX weights
    weights_path = model_path / "weights.npz"
    if weights_path.exists():
        weights = mx.load(str(weights_path))
    else:
        print(f"No weights file found in {weights_path}")
        exit(0)

    # Convert weights
    state = {}
    for k, v in tqdm.tqdm(weights.items(), total=len(weights), desc="Converting weights"):
        state[k] = torch.tensor(np.array(v))
    
    # Save weights in torch format
    torch_path = model_path / "consolidated.00.pth"
    torch.save(state, str(torch_path))