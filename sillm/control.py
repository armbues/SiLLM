import argparse
import re

import sillm
import sillm.utils as utils
import sillm.experimental.control as control

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="A simple CLI for generating text with SiLLM and control vectors.")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("vectors", type=str, help="The control vectors file")
    parser.add_argument("-a", "--alpha", type=float, default=1.0, help="Control vector scaling factor")
    parser.add_argument("-b", "--beta", type=float, default=-1.0, help="Projection vector scaling factor")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Seed for randomization")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("-p", "--repetition_penalty", type=float, default=None, help="Repetition penalty")
    parser.add_argument("-w", "--repetition_window", type=int, default=50, help="Window of generated tokens to consider for repetition penalty")
    parser.add_argument("-f", "--flush", type=int, default=5, help="Flush output every n tokens")
    parser.add_argument("-m", "--max_tokens", type=int, default=1024, help="Max. number of tokens to generate")
    parser.add_argument("--template", type=str, default=None, help="Chat template (chatml, llama2, alpaca, etc.)")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for chat template")
    parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity")
    args = parser.parse_args()
    
    # Initialize logging
    log_level = 40 - (10 * args.verbose) if args.verbose > 0 else 0
    logger = utils.init_logger(log_level)

    # Log commandline arguments
    if log_level <= 10:
        utils.log_arguments(args.__dict__)

    # Set random seed
    if args.seed >= 0:
        utils.seed(args.seed)

    # Load model
    model = sillm.load(args.model)

    # Initialize control model
    model = control.ControlledLLM.from_model(model)

    # Load control vectors
    model.load_control_vectors(args.vectors)

    generate_args = {
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "repetition_window": args.repetition_window,
        "max_tokens": args.max_tokens,
        "flush": args.flush
    }

    # Init conversation template
    template = sillm.init_template(model.tokenizer, model.args, args.template)
    conversation = sillm.Conversation(template, system_prompt=args.system_prompt)

    # Log memory usage
    utils.log_memory_usage()

    # Input loop
    while True:
        prompt = input("> ")

        if prompt.startswith('.'):
            break
        elif prompt == "":
            if conversation:
                conversation.clear()
            continue

        if re.match(r"^[\+\-]+$", prompt):
            alpha = 0.0
            for c in prompt:
                if c == "+":
                    alpha += 1.0
                elif c == "-":
                    alpha -= 1.0
            model.set_coeff(alpha=alpha)
            continue

        if conversation:
            prompt = conversation.add_user(prompt)
        
        logger.debug(f"Generating {args.max_tokens} tokens with temperature {args.temperature}")

        response = ""
        for s, metadata in model.generate(prompt, **generate_args):
            print(s, end="", flush=True)
            response += s
        print()

        if conversation:
            conversation.add_assistant(response)

        logger.debug(f"Evaluated {metadata['usage']['prompt_tokens']} prompt tokens in {metadata['timing']['eval_time']:.2f}s ({metadata['usage']['prompt_tokens'] / metadata['timing']['eval_time']:.2f} tok/sec)")
        logger.debug(f"Generated {metadata['usage']['completion_tokens']} tokens in {metadata['timing']['runtime']:.2f}s ({metadata['usage']['completion_tokens'] / metadata['timing']['runtime']:.2f} tok/sec)")