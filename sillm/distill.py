import argparse

import sillm
import sillm.utils as utils
from sillm.experimental.distillation import DistillationLoRA

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="A simple CLI for distilling a model with LoRA.")
    parser.add_argument("draft", type=str, help="The input model directory or file")
    parser.add_argument("target", type=str, help="The output model directory or file")
    parser.add_argument("-c", "--config", default=None, type=str, help="Load YAML configuration file for training")
    parser.add_argument("-t", "--train", default=None, type=str, help="Train the model with training dataset in the file/directory")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Seed for randomization")
    parser.add_argument("--max_entries", default=None, type=int, help="Max number of entries to load from training dataset (default: unlimited)")
    parser.add_argument("--max_length", default=1024, type=int, help="Max token length per training dataset entry (default: 1024)")
    parser.add_argument("--template", type=str, default=None, help="Chat template (chatml, llama2, alpaca, etc.)")
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
    parser.add_argument("--batch_size", default=1, type=int, help="Size of training batches (default: 4)")
    parser.add_argument("--report_steps", default=10, type=int, help="Number of batch iterations per training report (default: 10)")
    parser.add_argument("--eval_steps", default=100, type=int, help="Number of batch iterations per evaluation (default: 100)")
    parser.add_argument("--validation_samples", default=40, type=int, help="Number of validation_samples (default: 40)")
    parser.add_argument("--loss_alpha", default=0.5, type=float, help="Distillation loss alpha (default: 0.5)")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()

    # Load YAML configuration file
    if args.config is not None:
        utils.load_yaml(args.config, args)

    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)

    # Log commandline arguments
    if log_level <= 10:
        utils.log_arguments(args.__dict__)

    # Set random seed
    if args.seed >= 0:
        utils.seed(args.seed)

    # Load models
    draft_model = sillm.load(args.draft)
    target_model = sillm.load(args.target)

    # Freeze draft model
    draft_model.model.freeze()

    # Initialize trainable model
    distillation_config = {
        "loss_alpha": args.loss_alpha
    }
    target_model = DistillationLoRA.from_model(target_model, draft_model, **distillation_config)
    
    # Initialize LoRA layers
    lora_config = {
        "num_layers":       args.lora_layers,
        "target_modules":   args.lora_modules,
        "rank":             args.lora_rank,
        "dropout":          args.lora_dropout,
        "scale":            args.lora_scale
    }
    target_model.init_lora(**lora_config)

    # Set conversation template
    template = sillm.init_template(draft_model.tokenizer, draft_model.args, template_name=args.template)
    
    if args.train is not None:
        # Load training dataset
        dataset_config = {
            "template": template,
            "max_entries": args.max_entries,
            "max_length": args.max_length
        }
        dataset_training, dataset_validation, dataset_test = sillm.load_dataset(draft_model.tokenizer, args.train, **dataset_config)

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
        target_model.train(dataset_training,
                        dataset_validation,
                        **training_config)