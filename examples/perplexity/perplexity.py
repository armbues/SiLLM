import argparse

import sillm
import sillm.utils as utils

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Test models with the MMLU benchmark.")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("--batch_size", default=4, type=int, help="Size of training batches (default: 4)")
    parser.add_argument("-q4", default=False, action="store_true", help="Quantize the model to 4 bits")
    parser.add_argument("-q8", default=False, action="store_true", help="Quantize the model to 8 bits")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()

    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)

    # Load model
    model = sillm.load(args.model)

    # Quantize model
    if args.q4 is True:
        model.quantize(bits=4)
    elif args.q8 is True:
        model.quantize(bits=8)

    # Load Wikipedia sample dataset
    entries = sillm.training.dataset.load_jsonl("data/wikipedia.jsonl")
    dataset = sillm.DatasetCompletion(entries, model.tokenizer, max_length=1024)

    # Calculate perplexity
    num_batches = 0
    sum_batches = 0.0
    print(f"Computing perplexity on {len(dataset)} samples with batches of size {args.batch_size}:")
    for p in model.perplexity(dataset, batch_size=args.batch_size):
        num_batches += 1
        sum_batches += p
        avg = sum_batches / num_batches

        print(f"#{num_batches} perplexity: {p:.2f} (avg. {avg:.2f})")
    print(f"\nFinal avg. perplexity: {avg:.2f}")