import logging
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

import sillm.tokenizer
import sillm.args
import sillm.llama as llama
import sillm.gemma as gemma
import sillm.mixtral as mixtral

class LLM():
    """
    LLM model wrapper.
    """
    def __init__(self,
                 tokenizer: sillm.tokenizer.Tokenizer,
                 args: sillm.args.ModelArgs
                 ):
        """
        Args:
            tokenizer: Tokenizer instance.
            args: Model arguments.
        """
        self.args = args
        self._quantization = None

        if args.model_type in ("llama", "mistral"):
            self.model = llama.Model(args)
        elif args.model_type == "gemma":
            self.model = gemma.Model(args)
        elif args.model_type == "mixtral":
            self.model = mixtral.Model(args)
        else:
            raise NotImplementedError(f"Model type {args.model_type} is not supported")
        self.model.train(mode=False)
        self._update_names()

        self.tokenizer = tokenizer

    def _update_names(self):
        """
        Update module names.
        """
        for name, module in self.model.named_modules():
            module.name = name

    def update_weights(self,
                       weights: dict
                       ):
        """
        Update model weights.
        Args:
            weights: Weights to update.
        """
        weights = tree_unflatten(list(weights.items()))
        self.model.update(weights)
        mx.eval(self.model.parameters())

    def verify_weights(self,
                       weights: dict
                       ) -> bool:
        """
        Verify that weights for all modules are present and shapes match.
        Args:
            weights: Weights to verify.
        Returns:
            True if all weights are present and shapes match, False otherwise.
        """
        model_params = tree_flatten(self.model.parameters())
        result = True

        for name, weight in model_params:
            if name not in weights:
                result = False

                logging.warn(f"Key {name} not found in weights")
            elif weight.shape != weights[name].shape:
                result = False

                logging.warn(f"Shape mismatch for key {name}: {weight.shape} != {weights[name].shape}")

        model_keys = {name for name, _ in model_params}
        for name in weights:
            if name not in model_keys:
                logging.debug(f"Unused key {name} in weights")

        return result

    def save_weights(self,
                 weights_path: str
                 ):
        """
        Save model weights.
        Args:
            weights_path: Path to weights file.
        """
        self.model.save_weights(weights_path)

        logging.info(f"Saved model weights to {weights_path}")

    def quantize(self,
                 group_size: int = 32,
                 bits: int = 4,
                 excluded: list[str] = []
                 ):
        """
        Quantize model.
        Args:
            group_size: Group size for quantization.
            bits: Number of bits for quantization.
            excluded: List of module names to exclude from quantization.
        """
        if self._quantization is None:
            quantization = {
                "group_size": group_size,
                "bits": bits
            }
            self._quantization = quantization

            linear_class_predicate = lambda m: isinstance(m, nn.Linear) and m.weight.shape[0] != 8 and m.name not in excluded
            nn.QuantizedLinear.quantize_module(
                model = self.model,
                **quantization,
                linear_class_predicate = linear_class_predicate
            )

            self._update_names()

            logging.info(f"Quantized model with group size {group_size} and {bits} bits")
        else:
            logging.warn(f"Model is already quantized with group size {group_size} and {bits} bits")

    def dequantize(self):
        """
        Dequantize model.
        """
        if self._quantization is None:
            logging.warn(f"Model is not quantized")
        else:
            layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.QuantizedLinear):
                    bias = "bias" in module
                    weight = mx.dequantize(module.weight, module.scales, module.biases, module.group_size, module.bits).astype(mx.float16)
                    output_dims, input_dims = weight.shape
                    linear = nn.Linear(input_dims, output_dims, bias=bias)
                    linear.weight = weight
                    if bias:
                        linear.bias = module.bias

                    layers.append((name, linear))
            
            self.model.update_modules(tree_unflatten(layers))
            self._quantization = None

            self._update_names()

            logging.info(f"Dequantized model")

    def generate(self,
                 prompt: str,
                 temp: float = 0.0,
                 num_tokens: int = 256,
                 flush: int = 5
                 ):
        """
        Iterator for generating tokens.
        Args:
            prompt: Prompt to start generation.
            temp: Sampling temperature.
            num_tokens: Max number of tokens to generate.
            flush: Flush every `flush` tokens.
        Yields:
            Tuple of generated text and metadata.
        """
        yield from generate(self.model, self.tokenizer, prompt, temp, num_tokens, flush)

def generate(model,
             tokenizer: sillm.tokenizer.Tokenizer,
             prompt: str,
             temp: float = 0.0,
             num_tokens: int = 256,
             flush: int = 5
             ):
    """
    Iterator for generating tokens.
    Args:
        prompt: Prompt to start generation.
        temp: Sampling temperature.
        num_tokens: Max number of tokens to generate.
        flush: Flush every `flush` tokens.
    Yields:
        Tuple of generated text and metadata.
    """
    start = time.perf_counter()

    # Tokenize prompt
    inputs = mx.array(tokenizer.encode(prompt))

    # Initialize metadata
    metadata = {
        "runtime": 0.0,
        "eval_time": 0.0,
        "tokenizer_time": time.perf_counter() - start,
        "num_tokens": 0,
        "num_input": len(inputs)
    }

    def generate_step():
        def sample(logits):
            if temp > 0:
                return mx.random.categorical(logits * (1 / temp))
            else:
                return mx.argmax(logits, axis=-1)

        y = inputs
        cache = None
        while True:
            logits, cache = model(y[None], cache=cache)
            logits = logits[:, -1, :]
            y = sample(logits)

            yield y

    tokens = []
    for token, i in zip(generate_step(), range(num_tokens)):
        if i == 0:
            mx.eval(token)

            metadata["eval_time"] = time.perf_counter() - start

        if token[0] == tokenizer.eos_id:
            break

        tokens.append(token.item())

        if (len(tokens) % flush) == 0:
            mx.eval(token)
            s = tokenizer.decode(tokens)
            tokens = []

            metadata["num_tokens"] = i+1
            metadata["runtime"] = time.perf_counter() - start

            yield s, metadata

    mx.eval(token)
    s = tokenizer.decode(tokens)

    metadata["num_tokens"] = i+1
    metadata["runtime"] = time.perf_counter() - start

    yield s, metadata