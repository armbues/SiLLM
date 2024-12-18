import os
import argparse
import readline

import sillm
import sillm.utils as utils

if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="A simple CLI for generating text with SiLLM.")
    parser.add_argument("model", type=str, help="The model directory or file")
    parser.add_argument("-d", "--chdir", default=None, type=str, help="Change working directory")
    parser.add_argument("-c", "--config", default=None, type=str, help="Load YAML configuration file for chat")
    parser.add_argument("-a", "--input_adapters", default=None, type=str, help="Load and merge LoRA adapter weights from .safetensors file")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Seed for randomization")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("-k", "--top_k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("-p", "--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("-n", "--top_nsigma", type=float, default=0.0, help="Top-nsigma sampling")
    parser.add_argument("-r", "--repetition_penalty", type=float, default=None, help="Repetition penalty")
    parser.add_argument("-w", "--repetition_window", type=int, default=50, help="Window of generated tokens to consider for repetition penalty")
    parser.add_argument("-f", "--flush", type=int, default=5, help="Flush output every n tokens")
    parser.add_argument("-m", "--max_tokens", type=int, default=1024, help="Max. number of tokens to generate")
    parser.add_argument("-q", "--quantize", type=int, default=None, help="Quantize the model weights to the specified number of bits")
    parser.add_argument("--qkv", type=int, default=None, help="Quantize the KV cache to the specified number of bits")
    parser.add_argument("--cache", type=int, default=0, help="Prompt cache size")
    parser.add_argument("--template", type=str, default=None, help="Chat template (chatml, llama2, alpaca, etc.)")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for chat template")
    parser.add_argument("--ascii", default=False, action="store_true", help="Force output tokens to ASCII printable characters")
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

    # Set random seed
    if args.seed >= 0:
        utils.seed(args.seed)

    # Load model
    model = sillm.load(args.model)

    if args.input_adapters is not None:
        # Convert model to trainable
        model = sillm.TrainableLoRA.from_model(model)

        lora_config = model.load_lora_config(args.input_adapters)

        # Initialize LoRA layers
        model.init_lora(**lora_config)

        # Load and merge adapter file
        model.load_adapters(args.input_adapters)
        model.merge_and_unload_lora()

    # Quantize model
    if args.quantize is not None:
        model.quantize(bits=args.quantize)
    
    # Initialize prompt cache
    prompt_cache = None
    if args.cache > 0:
        prompt_cache = sillm.PromptCache(max_size=args.cache)

    # Initialize logit filters
    logit_filter = None
    if args.ascii:
        logit_filter = sillm.experimental.logit_filter.ASCIIFilter(model.tokenizer, model.args.vocab_size)

    generate_args = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "top_nsigma": args.top_nsigma,
        "repetition_penalty": args.repetition_penalty,
        "repetition_window": args.repetition_window,
        "max_tokens": args.max_tokens,
        "flush": args.flush,
        "prompt_cache": prompt_cache,
        "logit_filter": logit_filter
    }

    kv_cache_args = {
        "quantized": args.qkv is not None,
        "bits": args.qkv
    }

    # Initialize generator variables
    template = sillm.init_template(model.tokenizer, model.args, args.template)
    conversation = sillm.Conversation(template, system_prompt=args.system_prompt)
    cache = model.init_kv_cache(**kv_cache_args)

    # Log memory usage
    utils.log_memory_usage()

    # Input loop
    prompt = ""
    while True:
        prompt += input("> ")

        if prompt.startswith('/') or len(prompt) == 0:
            if prompt == "/exit":
                # Exit chat
                break
            elif prompt == "/clear":
                # Clear conversation
                conversation.clear()
                cache = model.init_kv_cache(**kv_cache_args)
            else:
                print("Commands:")
                print("/exit - Exit chat")
                print("/clear - Clear conversation")
            
            prompt = ""
            continue
        elif prompt.startswith('@'):
            # Load prompt from file
            fpath = prompt.lstrip('@')
            if not os.path.isfile(fpath):
                continue
            with open(fpath, "r") as f:
                prompt = f.read()
        elif prompt.endswith('\\'):
            # Continue prompt after line break
            prompt = prompt.rstrip('\\') + "\n"
            continue

        # Add user message to conversation and get prompt string
        request = conversation.add_user(prompt)

        logger.debug(f"Generating {args.max_tokens} tokens with temperature {args.temperature}")
        response = ""
        for s, metadata in model.generate(request, cache=cache, **generate_args):
            print(s, end="", flush=True)
            response += s
        print()

        conversation.add_assistant(response)
        prompt = ""

        logger.debug(f"Evaluated {metadata['usage']['prompt_tokens']} prompt tokens in {metadata['timing']['eval_time']:.2f}s ({metadata['usage']['prompt_tokens'] / metadata['timing']['eval_time']:.2f} tok/sec)")
        logger.debug(f"Generated {metadata['usage']['completion_tokens']} tokens in {metadata['timing']['runtime']:.2f}s ({metadata['usage']['completion_tokens'] / metadata['timing']['runtime']:.2f} tok/sec)")