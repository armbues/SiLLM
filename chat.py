import sys
import argparse
import logging

import mlx.core as mx

import sillm
import sillm.utils as utils

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="A simple CLI for generating text with SiLLM.")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Seed for randomization")
    parser.add_argument("-q4", default=False, action="store_true", help="Quantize the model to 4 bits")
    parser.add_argument("-q8", default=False, action="store_true", help="Quantize the model to 8 bits")
    parser.add_argument("-t", "--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("-f", "--flush", type=int, default=5, help="Flush output every n tokens")
    parser.add_argument("-n", "--num_tokens", type=int, default=512, help="Max. number of tokens to generate")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()
    
    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logging.basicConfig(level=log_level, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

    # Set random seed
    if args.seed >= 0:
        mx.random.seed(args.seed)

    # Load model
    model = sillm.load(args.model)

    # Quantize model
    if args.q4 is True:
        model.quantize(bits=4)
    elif args.q8 is True:
        model.quantize(bits=8)

    utils.log_memory_usage()

    # Input loop
    while True:
        prompt = input("> ")

        if prompt.startswith('.'):
            break
        
        logging.debug(f"Generating {args.num_tokens} tokens with temperature {args.temp}")

        for s, metadata in model.generate(prompt, temp=args.temp, num_tokens=args.num_tokens, flush=args.flush):
            print(s, end="", flush=True)
        print()

        logging.debug(f"Evaluated {metadata['num_input']} prompt tokens in {metadata['eval_time']:.2f}s ({metadata['num_input'] / metadata['eval_time']:.2f} tok/sec)")
        logging.debug(f"Generated {metadata['num_tokens']} tokens in {metadata['runtime']:.2f}s ({metadata['num_tokens'] / metadata['runtime']:.2f} tok/sec)")