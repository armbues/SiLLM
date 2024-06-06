import logging
import time
import pathlib
import json

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
    "phi3":         models.phi3.Model,
    "starcoder2":   models.starcoder2.Model,
    "qwen2":        models.qwen2.Model,
    "dbrx":         models.dbrx.Model,
    "cohere":       models.cohere.Model,
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
        self._mapping = None

        if args.model_type in model_map:
            model_class = model_map[args.model_type]
            self.model = model_class(args)
        else:
            raise NotImplementedError(f"Model type {args.model_type} is not supported")
        self.model.train(mode=False)

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

    def get_size(self):
        """
        Get model size.
        """
        total_size = 0
        for module in self.model.modules():
            if isinstance(module, nn.QuantizedLinear):
                total_size += module.weight.size * (32 // module.bits)
            elif isinstance(module, nn.Linear):
                total_size += module.weight.size

        return total_size

    def update_weights(self,
                       weights: dict,
                       mapping: dict = None
                       ):
        """
        Update model weights.
        Args:
            weights: Weights to update.
        """
        # Update weights
        weights = tree_unflatten(list(weights.items()))
        self.model.update(weights)
        mx.eval(self.model.parameters())

        # Add key mapping for potential quantization
        for key1 in list(mapping.keys()):
            key2 = mapping[key1]
            if key1.endswith(".weight"):
                mapping[key1.removesuffix(".weight") + ".scales"] = key2.removesuffix(".weight") + ".scales"
                mapping[key1.removesuffix(".weight") + ".biases"] = key2.removesuffix(".weight") + ".biases"
        
        # Update key mapping
        self._mapping = mapping

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
        Save model weights into a single safetensors file.
        Args:
            weights_path: Path to weights file.
        """
        state = dict(tree_flatten(self.model.parameters()))
        mx.save_safetensors(weights_path, state)

        logger.info(f"Saved model weights to {weights_path}")

    def save_shards(self,
                    weights_path: str,
                    max_shard_size: int = 5<<30
                    ):
        """
        Save model weights into shards.
        Args:
            weights_dir: Path to weights directory.
        """
        weights_path = pathlib.Path(weights_path)

        state = dict(tree_flatten(self.model.parameters()))

        shards = []
        weight_map = {}
        shard, shard_size = {}, 0
        total_size = 0
        for key, value in state.items():
            if self._mapping is not None:
                key = self._mapping[key]

            if shard_size + value.nbytes > max_shard_size:
                shards.append(shard)
                shard, shard_size = {}, 0

            shard[key] = value
            weight_map[key] = len(shards)

            shard_size += value.nbytes
            total_size += value.nbytes
        shards.append(shard)

        def save_shard(shard, shard_path):
            mx.save_safetensors(shard_path, shard)
            logger.debug(f"Saved shard to {shard_path}")

        if len(shards) > 1:
            for i, shard in enumerate(shards):
                save_shard(shard, str(weights_path / f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"))

            for key, i in weight_map.items():
                weight_map[key] = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        else:
            save_shard(shard, str(weights_path / "model.safetensors"))

            for key in weight_map:
                weight_map[key] = "model.safetensors"

        index_path = str(weights_path / "model.safetensors.index.json")
        with open(index_path, "w") as f:
            index_data = {
                "metadata": {
                    "total_size": total_size,
                },
                "weight_map": weight_map
            }

            f.write(json.dumps(index_data, indent=4))
            logger.debug(f"Saved weight index to {index_path}")

    def save(self,
             model_path: str,
             max_shard_size: int = 5<<30
             ):
        """
        Save model.
        Args:
            model_path: Path to model directory.
        """
        model_path = pathlib.Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model config
        config_path = model_path / "config.json"
        self.args.save_config(config_path)
        logger.debug(f"Saved model config to {config_path}")

        # Save tokenizer
        self.tokenizer.save(model_path)
        logger.debug(f"Saved tokenizer to {model_path}")

        # Save model weights
        self.save_shards(model_path, max_shard_size=max_shard_size)

        logger.info(f"Saved model to {model_path}")

    def quantize(self,
                 group_size: int = 32,
                 bits: int = 4,
                 weights: dict = None,
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
            self.args.quantization = quantization

            if weights is None:
                class_predicate = lambda p, m: (isinstance(m, (nn.Linear, nn.Embedding)) and
                                                ".gate." not in p
                                                )
            else:
                class_predicate = lambda p, m: (isinstance(m, (nn.Linear, nn.Embedding)) and
                                                f"{p}.scales" in weights
                                                )
            
            nn.quantize(self.model, class_predicate=class_predicate, **quantization)

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
                elif isinstance(module, nn.QuantizedEmbedding):
                    weight = mx.dequantize(module.weight, module.scales, module.biases, module.group_size, module.bits).astype(mx.float16)
                    num_embeddings, dims = weight.shape
                    embedding = nn.Embedding(num_embeddings, dims)
                    embedding.weight = weight
                    layers.append((name, embedding))
            
            if len(layers) > 0:
                self.model.update_modules(tree_unflatten(layers))

            self._quantization = None
            self.args.quantization = None

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

    def generate(self, *args, **kwargs):
        """
        Iterator for generating tokens.
        Args:
            prompt: Prompt to start generation.
            temperature: Sampling temperature.
            max_tokens: Max number of tokens to generate.
            repetition_penalty: Repetition penalty.
            repetition_window: Repetition window.
            logprobs: Return logprobs.
            flush: Flush buffer every n tokens.
        Yields:
            Tuple of generated text and metadata.
        """
        yield from generate(self.model, self.tokenizer, *args, **kwargs)

    def completion(self, *args, **kwargs) -> str:
        """
        Generate a completion and wait for all tokens.
        Args:
            prompt: Prompt to start generation.
            temperature: Sampling temperature.
            max_tokens: Max number of tokens to generate.
            repetition_penalty: Repetition penalty.
            repetition_window: Repetition window.
        Returns:
            Generated completion.
        """
        return ''.join([t[0] for t in generate(self.model, self.tokenizer, *args, **kwargs)])

def generate(model,
             tokenizer: Tokenizer,
             prompt: str,
             temperature: float = 0.0,
             max_tokens: int = 1024,
             repetition_penalty: float = None,
             repetition_window: int = 25,
             logprobs: bool = False,
             flush: int = 5
             ):
    start = time.perf_counter()

    # Tokenize prompt
    inputs = mx.array(tokenizer.encode(prompt))

    # Define stop tokens
    stop_tokens = tokenizer.special_ids
    
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

    # Initialize token and string buffers
    tokens, text = [], ""
    buffer, output = [], ""

    def sample(logits):
        if temperature > 0:
            return mx.random.categorical(logits * (1 / temperature))
        else:
            return mx.argmax(logits, axis=-1)
        # TODO add top-p sampling

    def apply_repetition_penalty(logits):
        indices = mx.array(tokens[-repetition_window:])
        repeated_logits = logits[:, indices]
        repeated_logits = mx.where(repeated_logits < 0, repeated_logits * repetition_penalty, repeated_logits / repetition_penalty)
        logits[:, indices] = repeated_logits

        return logits

    def generate_step(model, inputs):
        y = inputs
        cache = None
        while True:
            logits, cache = model(y[None], cache=cache)
            logits = logits[:, -1, :]

            if len(tokens) > 0 and repetition_penalty is not None:
                logits = apply_repetition_penalty(logits)

            y = sample(logits)

            p = 0.0
            if logprobs:
                p = nn.log_softmax(logits, axis=-1)[0,y].item()
            
            yield y, p

    # Main generation loop
    for (token,p), i in zip(generate_step(model, inputs), range(max_tokens)):
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
            mx.eval(tokens)

            text = tokenizer.decode(tokens)
            window = tokenizer.decode(buffer + tokens)
            if output + " " + text == window:
                output = text
                text = " " + text
            else:
                output = text
            buffer = tokens
            tokens = []

            timing["runtime"] = time.perf_counter() - start
            usage["completion_tokens"] = i+1
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            yield text, metadata

    text = tokenizer.decode(tokens)
    window = tokenizer.decode(buffer + tokens)
    if output + " " + text == window:
        output = text
        text = " " + text
    else:
        output = text

    timing["runtime"] = time.perf_counter() - start
    usage["completion_tokens"] = i+1
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    yield text, metadata