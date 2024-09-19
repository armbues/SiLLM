import logging
import pathlib
import json

import mlx.core as mx

from .mapping import map_key

logger = logging.getLogger("sillm")

def quantize_files(input_path: str,
                   output_path: str,
                   group_size: int = 32,
                   bits: int = 4
                   ):
    """
    Quantize weights files.
    Args:
        input_path: Path to load weights.
        output_path: Path to save quantized weights.
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    weights_files = sorted(list(input_path.glob("*.safetensors")))
    if len(weights_files) == 0:
        raise ValueError("No weights files found")
    
    def save_shard(shard, shard_path):
        mx.save_safetensors(shard_path, shard)
        logger.debug(f"Saved quantized shard to {shard_path}")
    
    shard = {}
    weight_map = {}
    original_size = 0
    total_size = 0
    for weights_path in weights_files:
        shard_path = output_path / weights_path.name
        if shard_path.exists():
            logger.debug(f"Skipping existing file: {shard_path}")
            continue

        logger.debug(f"Loading model weights file {weights_path}")

        weights = mx.load(str(weights_path))
        for key, weight in weights.items():
            original_size += weight.nbytes
            mlx_key = map_key(key)

            # Checking if weight should be quantized
            quantize_weight = key.endswith("weight") and len(weight.shape) > 1
            if weight.shape[0] == 8 or ".gate." in mlx_key:
                logger.debug(f"Skipping quantization for MoE gate: {key}")
                quantize_weight = False
            elif ".k_norm." in mlx_key or ".q_norm." in mlx_key:
                logger.debug(f"Skipping quantization for attention norm: {key}")
                quantize_weight = False
            elif len(weight.shape) > 1 and weight.shape[1] < 512:
                if bits == 4 and weight.shape[1] < 256:
                    logger.debug(f"Skipping quantization for small weight: {key}")
                    quantize_weight = False
                if bits == 8 and weight.shape[1] < 128:
                    logger.debug(f"Skipping quantization for small weight: {key}")
                    quantize_weight = False

            if quantize_weight:
                weight, scales, biases = mx.quantize(weight, group_size, bits)
                quant_size = weight.nbytes + scales.nbytes + biases.nbytes
                total_size += quant_size

                key_scales = key.replace(".weight", ".scales")
                key_biases = key.replace(".weight", ".biases")

                shard[key_scales] = scales
                shard[key_biases] = biases
            shard[key] = weight
            weight_map[key] = weights_path.name

        save_shard(shard, str(shard_path))
        shard = {}

    logger.debug(f"Quantization reduced weights size: {original_size//1024//1024:,} MB => {total_size//1024//1024:,} MB")

    index_path = str(output_path / "model.safetensors.index.json")
    with open(index_path, "w") as f:
        index_data = {
            "metadata": {
                "total_size": total_size,
            },
            "weight_map": weight_map
        }

        f.write(json.dumps(index_data, indent=4))
        logger.debug(f"Saved weight index to {index_path}")