import sys
import argparse
import pathlib
import logging

import mlx.core as mx

import sillm

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model", type=str, help="Directory for the base model")
    parser.add_argument("-i", "--load_adapter_path", default=None, type=str, help="Path to load adapter weights from .npz file")
    parser.add_argument("-a", "--save_adapter_path", default=None, type=str, help="Path to save adapter weights as .npz file")
    parser.add_argument("-o", "--save_merge_path", default=None, type=str, help="Path to save merged model weights as .npz file")
    parser.add_argument("-t", "--training_data", default=None, type=str, help="Train the model with training dataset in the directory")
    parser.add_argument("--layers", default=-1, type=int, help="Layers to use for LoRA (-1 for all layers)")
    parser.add_argument("--rank", default=8, type=int, help="Rank to use for LoRA")
    parser.add_argument("--iterations", default=1000, type=int, help="Number of iterations")
    parser.add_argument("--batch_size", default=4, type=int, help="Size of training batches")
    parser.add_argument("--seed", default=0, type=int, help="Seed for randomization")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()
    model_path = pathlib.Path(args.model)

    # Initialize logging
    log_level = 30 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(level=log_level, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

    # Load and init tokenizer/configuration/model and load the weights
    tokenizer = sillm.Tokenizer(str(model_path / "tokenizer.model"))
    model_args = sillm.ModelArgs.load(str(model_path / "config.json"))
    model = sillm.TrainableLLM(tokenizer, model_args)
    model.load_weights(model_path)
    
    if args.seed >= 0:
        mx.random.seed(args.seed)

    # Initialize LoRA layers
    model.init_lora(num_layers=args.layers, rank=args.rank)

    if args.load_adapter_path is not None:
        assert pathlib.Path(args.load_adapter).exists(), args.adapter_input

        # Load adapter file
        model.load_adapter(args.load_adapter)

    if args.training_data:
        # Load training dataset
        dataset_training, dataset_validation, dataset_test = sillm.Dataset.load(tokenizer, args.training_data)

        # Start training
        model.train(dataset_training, dataset_validation, batch_size=args.batch_size, iterations=args.iterations)

    if args.save_adapter_path:
        # Load adapter file
        model.save_adapters(args.save_adapter_path)

    if args.save_merge_path:
        # Merge LoRA layers back into model
        model.merge_and_unload_lora()

        # Save merged weights
        model.save_weights(args.save_merge_path)