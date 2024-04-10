import argparse

import sillm
import sillm.utils as utils

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="A simple CLI for generating text with SiLLM.")
    parser.add_argument("input", type=str, help="The input model directory or file")
    parser.add_argument("output", type=str, help="The output model directory or file")
    parser.add_argument("-a", "--input_adapters", default=None, type=str, help="Load LoRA adapter weights from .safetensors file")
    parser.add_argument("-r", "--remap", default=False, action="store_true", help="Remap weights keys to native SiLLM format")
    parser.add_argument("-q4", default=False, action="store_true", help="Quantize the model to 4 bits")
    parser.add_argument("-q8", default=False, action="store_true", help="Quantize the model to 8 bits")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()

    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)

    # Log commandline arguments
    if log_level <= 10:
        utils.log_arguments(args.__dict__)

    # Check for conversion arguments
    if args.input_adapters is None and args.q4 is False and args.q8 is False:
        raise ValueError("No conversion requested. Please specify adapters or quantization.")

    # Load model
    model = sillm.load(args.input)

    if args.input_adapters is not None:
        # Convert model to trainable
        model = sillm.TrainableLoRA.from_model(model)

        lora_config = model.load_lora_config(args.input_adapters)

        # Initialize LoRA layers
        model.init_lora(**lora_config)

        # Load and merge adapter file
        model.load_adapters(args.input_adapters)
        model.merge_and_unload_lora()

    # Quantize model
    if args.q4 is True:
        model.quantize(bits=4)
    elif args.q8 is True:
        model.quantize(bits=8)

    # Disable mapping to old keys
    if args.remap:
        model._mapping = None

    # Save model
    model.save(args.output)