import logging
import time
import pathlib

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .tokenizer import Tokenizer
import sillm.models as models
import sillm.models.args as args
from sillm.training.dataset import Dataset

logger = logging.getLogger("sillm")

model_map = {
    "llama":        models.llama.Model,
    "mistral":      models.llama.Model,
    "gemma":        models.gemma.Model,
    "mixtral":      models.mixtral.Model,   
    "phi":          models.phi.Model,
    "starcoder2":   models.starcoder2.Model,
    "qwen2":        models.qwen2.Model
}

class LLM():
    """
    LLM model wrapper.
    """
    def __init__(self,
                 tokenizer: Tokenizer,
                 args: args.ModelArgs
                 ):
        """
        Args:
            tokenizer: Tokenizer instance.
            args: Model arguments.
        """
        self.args = args
        self._quantization = None

        if args.model_type in model_map:
            model_class = model_map[args.model_type]
            self.model = model_class(args)
        else:
            raise NotImplementedError(f"Model type {args.model_type} is not supported")
        self.model.train(mode=False)
        self._update_names()

        self.tokenizer = tokenizer

    def init_description(self, model_path):
        """
        Set model description.
        Args:
            description: Model description.
        """
        self.path = pathlib.Path(model_path)
        self.id = self.path.name
        if self.path.is_file():
            self.id = self.path.stem    
        self.created = int(self.path.stat().st_ctime)
    
    def description(self):
        if self.id and self.created:
            return {
                "id": self.id,
                "object": "model",
                "created": self.created
            }
        else:
            raise ValueError("Model is missing an ID or creation time")
    
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

                logger.warn(f"Key {name} not found in weights")
            elif weight.shape != weights[name].shape:
                result = False

                logger.warn(f"Shape mismatch for key {name}: {weight.shape} != {weights[name].shape}")

        model_keys = {name for name, _ in model_params}
        for name in weights:
            if name not in model_keys:
                logger.debug(f"Unused key {name} in weights")

        return result

    def save_weights(self,
                     weights_path: str
                     ):
        """
        Save model weights.
        Args:
            weights_path: Path to weights file.
        """
        state = dict(tree_flatten(self.parameters()))
        metadata = {
            "format": "mlx"
        }
        mx.save_safetensors(weights_path, state, metadata=metadata)

        logger.info(f"Saved model weights to {weights_path}")

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

            logger.info(f"Quantized model with group size {group_size} and {bits} bits")
        else:
            logger.warn(f"Model is already quantized with group size {group_size} and {bits} bits")

    def dequantize(self):
        """
        Dequantize model.
        """
        if self._quantization is None:
            logger.warn(f"Model is not quantized")
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

            logger.info(f"Dequantized model")

    def perplexity(self,
                   dataset: Dataset,
                   batch_size: int = 4
                   ):
        """
        Calculate perplexity for an input text.
        Args:
            text: Input text.
        """
        for _, batch in zip(
            range(len(dataset)),
            dataset.iterate_batches(batch_size),
        ):
            losses, _, _ = self.model.loss(*batch)
        
            yield mx.exp(mx.mean(losses)).item()

    def generate(self,
                 prompt: str,
                 temperature: float = 0.0,
                 max_tokens: int = 1024,
                 flush: int = 5
                 ):
        """
        Iterator for generating tokens.
        Args:
            prompt: Prompt to start generation.
            temperature: Sampling temperature.
            max_tokens: Max number of tokens to generate.
            flush: Flush every `flush` tokens.
        Yields:
            Tuple of generated text and metadata.
        """
        yield from generate(self.model, self.tokenizer, prompt=prompt, temperature=temperature, max_tokens=max_tokens, flush=flush)

    def completion(self,
                   prompt: str,
                   temperature: float = 0.0,
                   max_tokens: int = 1024
                   ) -> str:
        """
        Generate a completion and wait for all tokens.
        Args:
            prompt: Prompt to start generation.
            temperature: Sampling temperature.
            max_tokens: Max number of tokens to generate.
        Returns:
            Generated completion.
        """
        return ''.join([t[0] for t in generate(self.model, self.tokenizer, prompt=prompt, temperature=temperature, max_tokens=max_tokens)])

def generate(model,
             tokenizer: Tokenizer,
             prompt: str,
             temperature: float = 0.0,
             max_tokens: int = 1024,
             logprobs: bool = False,
             flush: int = 5
             ):
    start = time.perf_counter()

    # Tokenize prompt
    inputs = mx.array(tokenizer.encode(prompt))

    # Define stop tokens
    stop_tokens = [tokenizer.eos_id, tokenizer.bos_id]
    
    # Initialize metadata
    timing = {
        "runtime": 0.0,
        "eval_time": 0.0,
        "tokenizer_time": time.perf_counter() - start
    }
    usage = {
        "prompt_tokens": len(inputs),
        "completion_tokens": 0,
        "total_tokens": 0
    }
    metadata = {
        "timing": timing,
        "usage": usage,
        "logprobs": [],
        "finish_reason": "length"
    }

    tokens = []
    for (token,p) , i in zip(generate_step(model, inputs, temperature, logprobs), range(max_tokens)):
        if i == 0:
            mx.eval(token)

            timing["eval_time"] = time.perf_counter() - start

        if token.item() in stop_tokens:
            metadata["finish_reason"] = "stop"
            break

        tokens.append(token.item())

        if logprobs:
            metadata["logprobs"].append(p)

        if (len(tokens) % flush) == 0:
            mx.eval(token)
            s = tokenizer.decode(tokens)
            tokens = []

            timing["runtime"] = time.perf_counter() - start
            usage["completion_tokens"] = i+1
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            yield s, metadata

    mx.eval(token)
    s = tokenizer.decode(tokens)

    timing["runtime"] = time.perf_counter() - start
    usage["completion_tokens"] = i+1
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    yield s, metadata

def generate_step(model, inputs, temperature, logprobs=False):
    y = inputs
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits, temperature)

        p = 0.0
        if logprobs:
            p = nn.log_softmax(logits, axis=-1)[0,y].item()
        
        yield y, p

def sample(logits, temperature):
    if temperature > 0:
        return mx.random.categorical(logits * (1 / temperature))
    else:
        return mx.argmax(logits, axis=-1)
    # TODO add top-p sampling