import argparse
import pathlib
import shutil

import sillm.utils as utils

from sillm.utils.quantization import quantize_files
from sillm.models.args import ModelArgs
from sillm.core.tokenizer import TransformerTokenizer, SentencePieceTokenizer

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="A simple CLI for generating text with SiLLM.")
    parser.add_argument("input", type=str, help="The input model directory or file")
    parser.add_argument("output", type=str, help="The output model directory or file")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits")
    parser.add_argument("--group_size", default=32, help="Quantization group size")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()

    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)

    # Log commandline arguments
    if log_level <= 10:
        utils.log_arguments(args.__dict__)

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    quantization = {
        "bits": args.bits,
        "group_size": args.group_size
    }

    quantize_files(args.input, args.output, **quantization)

    config_path = input_path / "config.json"
    model_args = ModelArgs.load_file(config_path)
    model_args.quantization = quantization
    config_path = output_path / "config.json"
    model_args.save_config(config_path)
    logger.debug(f"Saved model config to {config_path}")

    generation_config_path = input_path / "generation_config.json"
    if generation_config_path.exists():
        shutil.copy(generation_config_path, output_path / "generation_config.json")
        logger.debug(f"Copied generation config to {output_path}")

    tokenizer = None
    tokenizer_path = None
    if (input_path / "tokenizer.json").exists():
        tokenizer_path = input_path / "tokenizer.json"
        tokenizer = TransformerTokenizer(str(input_path), model_args)
    elif (input_path / "tokenizer.model").exists():
        tokenizer_path = input_path / "tokenizer.model"
        tokenizer = SentencePieceTokenizer(str(tokenizer_path), model_args)
    
    if tokenizer is None:
        logger.error(f"No tokenizer found in {input_path}")
        
    tokenizer.save(str(output_path))
    logger.debug(f"Saved tokenizer to {output_path}")

    logger.info(f"Quantized model with group size {args.group_size} and {args.bits} bits")