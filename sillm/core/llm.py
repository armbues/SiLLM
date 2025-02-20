import logging
import time
import pathlib
import json
import typing

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .tokenizer import Tokenizer
import sillm.models as models
import sillm.models.args as args
import sillm.utils.sampling as sampling
from sillm.training.dataset import Dataset
from sillm.core.cache import KVCache, QuantizedKVCache, PromptCache
from sillm.modules.switch import SwitchLinear
from sillm.experimental.logit_filter import LogitFilter

logger = logging.getLogger("sillm")

model_map = {
    "llama":        models.llama.Model,
    "mistral":      models.llama.Model,
    "gemma":        models.gemma.Model,
    "gemma2":       models.gemma2.Model,
    "mixtral":      models.mixtral.Model,   
    "phi":          models.phi.Model,
    "phi3":         models.phi3.Model,
    "phimoe":       models.phimoe.Model,
    "starcoder2":   models.starcoder2.Model,
    "qwen2":        models.qwen2.Model,
    "dbrx":         models.dbrx.Model,
    "cohere":       models.cohere.Model,
    "nemotron":     models.nemotron.Model,
    "pharia-v1":    models.pharia1.Model,
    "granite":      models.granite.Model,
    "mobilellm":    models.llama.Model,
    "exaone":       models.llama.Model,
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

        self._update_names()

        self.tokenizer = tokenizer

    def init_kv_cache(self,
                      step: int = 256,
                      quantized: bool = False,
                      group_size: int = 32,
                      bits: int = 4,
                      ):
        """
        Initialize KV cache for this model type.
        """
        if quantized:
            logger.debug(f"Initializing quantized KV cache ({bits} bits / group size {group_size}) with step size {step}")

            return QuantizedKVCache.for_model(self, step, group_size, bits)
        else:
            logger.debug(f"Initializing KV cache with step size {step}")

            return KVCache.for_model(self, step)

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

    @property
    def size(self):
        """
        Get model size in bytes.
        """
        total_size = 0
        for module in self.model.modules():
            if isinstance(module, nn.QuantizedLinear):
                total_size += int(module.weight.size * 32 / module.bits)
            elif isinstance(module, nn.Linear):
                total_size += module.weight.size

        return total_size
    
    @property
    def max_tokens(self):
        """
        Get maximum number of tokens.
        """
        return self.args.max_position_embeddings
    
    def _update_names(self):
        """
        Update module names.
        """
        for name, module in self.model.named_modules():
            module._name = name
    
    def preprocess_weights(self,
                           weights: dict
                           ) -> dict:
        """
        Preprocess model weights.
        """
        if hasattr(self.model, "preprocess_weights"):
            logger.debug(f"Preprocessing model weights")

            return self.model.preprocess_weights(weights)
        
        return weights

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

    def astype(self,
               dtype: str
               ):
        """
        Cast model weights of linear modules to a different data type.
        """
        if dtype not in ("float16", "float32", "bfloat16"):
            raise ValueError(f"Invalid data type {dtype}")
        
        dtype = getattr(mx, dtype)
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.dtype != dtype:
                module.weight = module.weight.astype(dtype)

        logger.info(f"Cast model weights to {dtype}")
        
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
            weights: Keys to initialize quantization for existing weights.
        """
        if self._quantization is None:
            quantization = {
                "group_size": group_size,
                "bits": bits
            }
            self._quantization = quantization
            self.args.quantization = quantization

            if weights is None:
                class_predicate = lambda p, m: (isinstance(m, (nn.Linear, nn.Embedding, SwitchLinear)) and
                                                ".gate." not in p
                                                )
            else:
                class_predicate = lambda p, m: (isinstance(m, (nn.Linear, nn.Embedding, SwitchLinear)) and
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
            token_ids: Return token IDs.
            flush: Flush buffer every n tokens.
            extra_stop_tokens: Additional stop tokens.
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
            logprobs: Return logprobs.
            token_ids: Return token IDs.
            flush: Flush buffer every n tokens.
            extra_stop_tokens: Additional stop tokens.
        Returns:
            Tuple of generated text and metadata.
        """
        result = ""
        for text, metadata in generate(self.model, self.tokenizer, *args, **kwargs):
            result += text

        return result, metadata

def generate(model,
             tokenizer: Tokenizer,
             prompt: str | list | mx.array,
             cache: KVCache = None,
             max_tokens: int = 2048,
             temperature: float = 0.0,
             top_k: int = 0,
             top_p: float = 1.0,
             min_p: float = 0.0,
             top_nsigma: float = 0.0,
             repetition_penalty: float = None,
             repetition_window: int = 25,
             logprobs: bool = False,
             token_ids: bool = False,
             flush: int = 5,
             extra_stop_tokens: list = None,
             prompt_cache: PromptCache = None,
             logit_filter: LogitFilter = None
             ):
    start = time.perf_counter()

    # Pre-process inputs
    if isinstance(prompt, str):
        inputs = mx.array(tokenizer.encode(prompt))
    elif isinstance(prompt, list):
        inputs = mx.array(prompt)
    elif isinstance(prompt, mx.array):
        inputs = prompt
    else:
        raise ValueError("Prompt must be a string, list of tokens, or MX array")
    
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
        "token_ids": [],
        "finish_reason": "length"
    }

    # Define stop tokens
    stop_tokens = tokenizer.special_ids
    
    # Add user-defined stop tokens
    if extra_stop_tokens is not None:
        for token in extra_stop_tokens:
            if isinstance(token, str):
                token_ids = tokenizer.encode(token, bos=False)

                if len(token_ids) > 1:
                    logger.warning(f"Extra stop token '{token}' tokenizes to multiple tokens")
                stop_tokens.add(token_ids[0])
            else:
                stop_tokens.add(token)

    # Initialize token and string buffers
    tokens, text = [], ""

    def sample(logits):
        if temperature == 0:
            y = mx.argmax(logits, axis=-1)
        else:
            # Apply temperature
            logits = logits * (1 / temperature)

            # Apply structure enforcer
            if logit_filter is not None:
                logits = logit_filter(logits)
            # Apply repetition penalty
            if len(tokens) > 0 and repetition_penalty is not None:
                logits = sampling.apply_repetition_penalty(logits, tokens, repetition_penalty=repetition_penalty, repetition_window=repetition_window)
            # Apply top-k sampling
            if top_k > 0:
                logits = sampling.top_k(logits, k=top_k)
            elif 0.0 < top_p < 1.0:
                logits = sampling.top_p(logits, p=top_p)
            elif min_p > 0.0:
                logits = sampling.min_p(logits, p=min_p)
            elif top_nsigma > 0.0:
                logits = sampling.top_nsigma(logits, n=top_nsigma)
    
            y = mx.random.categorical(logits)

        p = 0.0
        if logprobs:
            p = nn.log_softmax(logits, axis=-1)[0,y].item()

        return y, p

    def generate_step(model, inputs):
        nonlocal cache
        logits = None

        if prompt_cache is not None:
            # Retrieve cached logits and KV cache
            logits, cache = prompt_cache.get(inputs)
        
        if cache is None:
            # Initialize KV cache
            cache = KVCache.for_model(model)

        if logits is None:
            # Initial forward pass through model
            logits = model(inputs[None], cache=cache)
            logits = logits[:, -1, :]

            # Store logits and KV cache in prompt cache
            if prompt_cache is not None:
                prompt_cache.put(inputs, logits, cache)

        y, p = sample(logits)
        yield y, p

        while True:
            # Iterative forward pass through model
            logits = model(y[None], cache=cache)
            logits = logits[:, -1, :]

            y, p = sample(logits)
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

            text_offset = len(text)
            text = tokenizer.decode(tokens)

            timing["runtime"] = time.perf_counter() - start
            usage["completion_tokens"] = len(tokens)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            if token_ids:
                metadata["token_ids"] += tokens

            yield text[text_offset:], metadata

    mx.async_eval(tokens)

    text_offset = len(text)
    text = tokenizer.decode(tokens)

    timing["runtime"] = time.perf_counter() - start
    usage["completion_tokens"] = i+1
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    if token_ids:
        metadata["token_ids"] = tokens

    yield text[text_offset:], metadata