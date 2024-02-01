import sys
import argparse
import logging

import mlx.core as mx

import sillm
import sillm.utils as utils

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Interface for training SiLLM models with LoRA/QLoRA.")
    parser.add_argument("model", type=str, help="The directory or file for the base model (MLX, Torch, GGUF)")
    parser.add_argument("-i", "--load_adapter_path", default=None, type=str, help="Load adapter weights from file (.npz)")
    parser.add_argument("-a", "--save_adapter_path", default=None, type=str, help="Save adapter weights to file (.npz)")
    parser.add_argument("-o", "--save_merge_path", default=None, type=str, help="Folder to save merged model weights")
    parser.add_argument("-t", "--training_data", default=None, type=str, help="Train the model with training dataset in the directory")
    parser.add_argument("-q", "--quantize", default=None, type=int, help="Quantize the model to the given number of bits")
    parser.add_argument("--layers", default=-1, type=int, help="Layers to use for LoRA (default: -1 for all layers)")
    parser.add_argument("--rank", default=8, type=int, help="Rank to use for LoRA (default: 8)")
    parser.add_argument("--iterations", default=1000, type=int, help="Number of iterations (default: 1000)")
    parser.add_argument("--rate", default=1e-5, type=float, help="Learning rate (default: 1e-5)")
    parser.add_argument("--batch_size", default=4, type=int, help="Size of training batches (default: 4)")
    parser.add_argument("--seed", default=0, type=int, help="Seed for randomization (default: 0)")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()
    
    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(level=log_level, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

    # Load model
    model = sillm.load(args.model)
    utils.log_memory_usage()
    
    if args.seed >= 0:
        mx.random.seed(args.seed)

    if args.quantize is not None:
        # Quantize model
        model.quantize(bits=args.quantize)
        utils.log_memory_usage()

    # Initialize LoRA layers
    model.init_lora(num_layers=args.layers, rank=args.rank)

    if args.load_adapter_path is not None:
        # Load adapter file
        model.load_adapters(args.load_adapter)

    if args.training_data:
        # Load training dataset
        dataset_training, dataset_validation, dataset_test = sillm.Dataset.load(model.tokenizer, args.training_data)

        # Start training
        model.train(dataset_training, dataset_validation, batch_size=args.batch_size, iterations=args.iterations)

    if args.save_adapter_path:
        # Save adapter file
        model.save_adapters(args.save_adapter_path)

    if args.save_merge_path:
        # Merge LoRA layers back into model
        model.merge_and_unload_lora()

        # Save merged weights
        model.save_npz(args.save_merge_path)
        # TODO save sharded weights