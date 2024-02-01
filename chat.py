import sys
import argparse
import logging

import mlx.core as mx

import sillm
import sillm.utils as utils

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("-q", "--quantize", default=None, type=int, help="Quantize the model to the given number of bits")
    parser.add_argument("-t", "--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Seed for randomization")
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
    utils.log_memory_usage()

    if args.quantize is not None:
        model.quantize(bits=args.quantize)
        utils.log_memory_usage()
    
    while True:
        prompt = input("> ")

        if prompt.startswith('.'):
            break
        
        for result, stats in model.generate(prompt, temp=args.temp, num_tokens=args.num_tokens):
            print(result, end="", flush=True)
        print()

        logging.info(f"Generated {stats['num_tokens']} tokens in {stats['runtime']:.2f}s ({stats['num_tokens'] / stats['runtime']:.2f} tok/sec)")