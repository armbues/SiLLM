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
    parser.add_argument("-i", "--load_adapters", default=None, type=str, help="Load adapter weights from file (.safetensors or .npz)")
    parser.add_argument("-o", "--save_adapters", default=None, type=str, help="Save adapter weights to file (.safetensors or .npz)")
    parser.add_argument("-c", "--save_checkpoints", default=None, type=str, help="Save model checkpoints to directory")
    parser.add_argument("-m", "--save_merge", default=None, type=str, help="Save merged model weights to file (.safetensors or .npz)")
    parser.add_argument("-d", "--data", default=None, type=str, help="Train the model with training dataset in the directory")
    parser.add_argument("--layers", default=-1, type=int, help="Layers to use for LoRA (default: -1 for all layers)")
    parser.add_argument("--rank", default=8, type=int, help="Rank to use for LoRA (default: 8)")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs (default: 1)")
    parser.add_argument("--iterations", default=0, type=int, help="Number of iterations per epoch (default: dataset size)")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate (default: 1e-5)")
    parser.add_argument("--batch_size", default=4, type=int, help="Size of training batches (default: 4)")
    parser.add_argument("--seed", default=0, type=int, help="Seed for randomization (default: 0)")
    parser.add_argument("-q4", default=False, action="store_true", help="Quantize the model to 4 bits")
    parser.add_argument("-q8", default=False, action="store_true", help="Quantize the model to 8 bits")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()
    
    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(level=log_level, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

    # Load base model
    model = sillm.load(args.model)
    
    # Set random seed
    if args.seed >= 0:
        mx.random.seed(args.seed)

    # Quantize model
    if args.q4 is True:
        model.quantize(bits=4)
    elif args.q8 is True:
        model.quantize(bits=8)

    # Initialize LoRA layers
    model = sillm.TrainableLoRA.from_model(model)
    model.init_lora(num_layers=args.layers, rank=args.rank)

    # Log memory usage
    utils.log_memory_usage()

    if args.load_adapters is not None:
        # Load adapter file
        model.load_adapters(args.load_adapters)

    if args.data:
        # Load training dataset
        dataset_training, dataset_validation, dataset_test = sillm.Dataset.load(model.tokenizer, args.data)

        def eval_callback(i, val_loss):
            if args.save_checkpoints is not None:
                model.save_checkpoint(args.save_checkpoints, i)

        # Start training
        model.train(dataset_training,
                    dataset_validation,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                    iterations=args.iterations,
                    eval_callback=eval_callback)

    if args.save_adapters:
        # Save adapter file
        model.save_adapters(args.save_adapters)

    if args.save_merge:
        # Merge LoRA layers back into model
        model.merge_and_unload_lora()

        # Save merged weights
        model.save_weights(args.save_merge)
        # TODO save sharded weights