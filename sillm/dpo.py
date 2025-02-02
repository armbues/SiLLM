import os
import argparse

import mlx.core as mx

import sillm
import sillm.utils as utils

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Interface for training SiLLM models with LoRA/QLoRA.")
    parser.add_argument("model", type=str, help="The directory or file for the base model (MLX, Torch, GGUF)")
    parser.add_argument("-d", "--chdir", default=None, type=str, help="Change working directory")
    parser.add_argument("-c", "--config", default=None, type=str, help="Load YAML configuration file for training")
    parser.add_argument("-t", "--train", default=None, type=str, help="Train the model with training dataset in the file/directory")
    parser.add_argument("-a", "--input_adapters", default=None, type=str, help="Load adapter weights from .safetensors file")
    parser.add_argument("-o", "--output_dir", default=None, type=str, help="Output directory for adapter weights")
    parser.add_argument("-s", "--save_checkpoints", default=False, action="store_true", help="Save model checkpoints to output directory")
    parser.add_argument("-m", "--save_merge", default=None, type=str, help="Save merged model weights to .safetensors file")
    parser.add_argument("-g", "--grad_checkpoint", default=False, action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--max_length", default=1024, type=int, help="Max token length per training dataset entry (default: 1024)")
    parser.add_argument("--template", type=str, default=None, help="Chat template (chatml, llama-2, alpaca, etc.)")
    parser.add_argument("--layers", default=-1, type=int, help="Layers to use for LoRA (default: -1 for all layers)")
    parser.add_argument("--target_modules", default="query_value", type=str, help="Target modules to use for LoRA: query_value, all_linear")
    parser.add_argument("--rank", default=8, type=int, help="Rank to use for LoRA (default: 8)")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate (default: 1e-5)")
    parser.add_argument("--loss_type", default="sigmoid", type=str, help="Loss type for DPO (default: sigmoid)")
    parser.add_argument("--loss_beta", default=0.1, type=float, help="Beta parameter for loss function (default: 0.1)")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs (default: 1)")
    parser.add_argument("--iterations", default=0, type=int, help="Number of iterations per epoch (default: dataset size)")
    parser.add_argument("--batch_size", default=4, type=int, help="Size of training batches (default: 4)")
    parser.add_argument("--report_steps", default=10, type=int, help="Number of batch iterations per training report (default: 10)")
    parser.add_argument("--eval_steps", default=100, type=int, help="Number of batch iterations per evaluation (default: 100)")
    parser.add_argument("--validation_samples", default=20, type=int, help="Number of validation_samples (default: 20)")
    parser.add_argument("--seed", default=0, type=int, help="Seed for randomization (default: 0)")
    parser.add_argument("--plot", default=None, type=str, help="Create a loss plot and save it to the specified file")
    parser.add_argument("-q4", default=False, action="store_true", help="Quantize the model to 4 bits")
    parser.add_argument("-q8", default=False, action="store_true", help="Quantize the model to 8 bits")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    parser.add_argument("--wandb", default=False, action="store_true", help="Log training to Weights & Biases")
    parser.add_argument("--wandb_project", default=None, type=str, help="Weights & Biases project name")
    parser.add_argument("--wandb_report_steps", default=5, type=int, help="Number of batch iterations per Weights & Biases report (default: 5)")
    args = parser.parse_args()

    # Change working directory
    if args.chdir is not None:
        os.chdir(args.chdir)

    # Load YAML configuration file
    if args.config is not None:
        utils.load_yaml(args.config, args)
    
    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)

    # Log commandline arguments
    if log_level <= 10:
        utils.log_arguments(args.__dict__)

    # Load base model
    model = sillm.load(args.model)
    
    # Set random seed
    if args.seed >= 0:
        utils.seed(args.seed)

    # Quantize model
    if args.q4 is True:
        model.quantize(bits=4)
    elif args.q8 is True:
        model.quantize(bits=8)

    # Initialize trainable model
    dpo_config = {
        "reference_free":   args.input_adapters is None,
        "loss_type":        args.loss_type,
        "loss_beta":        args.loss_beta
    }
    model = sillm.TrainableDPO.from_model(model, **dpo_config)
    
    # Initialize LoRA layers
    lora_config = {
        "num_layers":       args.layers,
        "target_modules":   args.target_modules,
        "rank":             args.rank
    }
    model.init_lora(**lora_config)

    if args.input_adapters is not None:
        # Load adapter file
        model.load_adapters(args.input_adapters)

    # Initialize plot
    if args.plot is not None:
        plot = utils.Plot()

    # Set conversation template
    if args.template:
        template = sillm.Template(model.tokenizer, template=args.template)
    else:
        template = None
    
    # Log memory usage
    logger.debug(f"Peak memory usage: {(mx.metal.get_peak_memory() // (1024 ** 2)):,} MB")

    if args.train is not None:
        # Load training dataset
        dataset_config = {
            "template": template,
            "max_length": args.max_length
        }
        dataset_training, dataset_validation, dataset_test = sillm.load_dataset(model.tokenizer, args.train, **dataset_config)

        def report_callback(i, loss):
            if args.plot is not None:
                plot.add_train_loss(i, loss)

        def eval_callback(i, val_loss):
            if args.plot is not None:
                plot.add_valid_loss(i, val_loss)
                plot.save(args.plot)
        
            if i > 1 and args.save_checkpoints and args.output_dir is not None:
                fpath_ckpt = model.save_checkpoint(args.output_dir, i)
                
                return f"Saved checkpoint to {fpath_ckpt}"

        # Model training
        training_config = {
            "grad_checkpoint":      args.grad_checkpoint,
            "batch_size":           args.batch_size,
            "learning_rate":        args.learning_rate,
            "epochs":               args.epochs,
            "iterations":           args.iterations,
            "report_steps":         args.report_steps,
            "eval_steps":           args.eval_steps,
            "validation_samples":   args.validation_samples,
            "wandb":                args.wandb,
            "wandb_project":        args.wandb_project,
            "wandb_report_steps":   args.wandb_report_steps,
        }
        model.train(dataset_training,
                    dataset_validation,
                    dataset_test,
                    eval_callback=eval_callback,
                    **training_config)
        
    if args.plot is not None:
        plot.save(args.plot)

    if args.save_checkpoints:
        # Save final checkpoint
        model.save_checkpoint(args.output_dir)

    if args.save_merge:
        # Merge LoRA layers back into model
        model.merge_and_unload_lora()

        # Save merged weights
        model.save_weights(args.save_merge)