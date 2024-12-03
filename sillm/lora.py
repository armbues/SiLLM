import os
import argparse

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
    parser.add_argument("--max_entries", default=None, type=int, help="Max number of entries to load from training dataset (default: unlimited)")
    parser.add_argument("--max_length", default=1024, type=int, help="Max token length per training dataset entry (default: 1024)")
    parser.add_argument("--template", type=str, default=None, help="Chat template (chatml, llama-2, alpaca, etc.)")
    parser.add_argument("--lora_layers", default=0, type=int, help="Layers to use for LoRA (default: 0 for all layers)")
    parser.add_argument("--lora_modules", default="query_value", type=str, help="Target modules to use for LoRA: query_value, all_linear")
    parser.add_argument("--lora_rank", default=8, type=int, help="Rank to use for LoRA (default: 8)")
    parser.add_argument("--lora_dropout", default=0.0, type=int, help="Dropout to use for LoRA (default: 0.0)")
    parser.add_argument("--lora_scale", default=10.0, type=float, help="Scale to use for LoRA (default: 10.0)")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer type (default: adam)")
    parser.add_argument("--grad_checkpoint", default=False, action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--grad_accu_steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate (default: 1e-5)")
    parser.add_argument("--learning_decay", default=0.0, type=float, help="Learning decay for optimizer schedule (default: 0.0)")
    parser.add_argument("--learning_warmup", default=0, type=int, help="Learning warmup for optimizer schedule (default: 0)")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs (default: 1)")
    parser.add_argument("--iterations", default=0, type=int, help="Number of iterations per epoch (default: dataset size)")
    parser.add_argument("--batch_size", default=4, type=int, help="Size of training batches (default: 4)")
    parser.add_argument("--report_steps", default=10, type=int, help="Number of batch iterations per training report (default: 10)")
    parser.add_argument("--eval_steps", default=100, type=int, help="Number of batch iterations per evaluation (default: 100)")
    parser.add_argument("--validation_samples", default=40, type=int, help="Number of validation_samples (default: 40)")
    parser.add_argument("--seed", default=0, type=int, help="Seed for randomization (default: 0)")
    parser.add_argument("--plot", default=None, type=str, help="Create a loss plot and save it to the specified file")
    parser.add_argument("--memory_limit", default=None, type=float, help="Memory limit for training (default: None)")
    parser.add_argument("--relax_memory_limit", default=False, action="store_true", help="Relax memory limit for training")
    parser.add_argument("-q4", default=False, action="store_true", help="Quantize the model to 4 bits")
    parser.add_argument("-q8", default=False, action="store_true", help="Quantize the model to 8 bits")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
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

    # Set memory limit
    if args.memory_limit is not None:
        utils.set_memory_limit(args.memory_limit, relaxed=args.relax_memory_limit)

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
    model = sillm.TrainableLoRA.from_model(model)
    
    # Initialize LoRA layers
    lora_config = {
        "num_layers":       args.lora_layers,
        "target_modules":   args.lora_modules,
        "rank":             args.lora_rank,
        "dropout":          args.lora_dropout,
        "scale":            args.lora_scale
    }
    model.init_lora(**lora_config)

    if args.input_adapters is not None:
        # Load adapter file
        model.load_adapters(args.input_adapters)

    # Initialize plot
    if args.plot is not None:
        plot = utils.Plot()

    # Set conversation template
    template = sillm.init_template(model.tokenizer, model.args, args.template)

    # Log memory usage
    utils.log_memory_usage()

    if args.train is not None:
        # Load training dataset
        dataset_config = {
            "template": template,
            "max_entries": args.max_entries,
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
            
        if args.output_dir is not None:
            model.save_lora_config(args.output_dir)

        # Model training
        training_config = {
            "batch_size":                   args.batch_size,
            "optimizer_type":               args.optimizer,
            "learning_rate":                args.learning_rate,
            "learning_decay":               args.learning_decay,
            "learning_warmup":              args.learning_warmup,
            "gradient_checkpointing":       args.grad_checkpoint,
            "gradient_accumulation_steps":  args.grad_accu_steps,
            "epochs":                       args.epochs,
            "iterations":                   args.iterations,
            "report_steps":                 args.report_steps,
            "eval_steps":                   args.eval_steps,
            "validation_samples":           args.validation_samples,
        }
        model.train(dataset_training,
                    dataset_validation,
                    report_callback=report_callback,
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