import pathlib

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

import sillm.model as model
import sillm.llama as llama
import sillm.llama as mixtral

class LLM():
    """
    LLM model wrapper.
    """
    def __init__(self, tokenizer, args: model.ModelArgs):
        """
        Args:
            tokenizer: Tokenizer instance.
            args: Model arguments.
        """
        self.args = args

        if args.model_type == "llama":
            self.model = llama.Model(args)
        elif args.model_type == "mistral":
            self.model = llama.Model(args)
        elif args.model_type == "mixtral":
            self.model = mixtral.Model(args)
        else:
            raise NotImplementedError(f"Model type {args.model_type} is not supported")
        self.tokenizer = tokenizer

        self.eos_id = tokenizer.eos_id
        self._quantization = None

    def load_weights(self, weights_path: str):
        """
        Load model weights.
        Args:
            weights_path: Path to weights file.
        """
        assert pathlib.Path(weights_path).exists(), weights_path

        weights = mx.load(weights_path)
        weights = tree_unflatten(list(weights.items()))
        self.model.update(weights)

        mx.eval(self.model.parameters())

    def save_weights(self, weights_path: str):
        """
        Save model weights.
        Args:
            weights_path: Path to weights file.
        """
        self.model.save_weights(weights_path)

    def quantize(self, group_size=64, bits=4):
        """
        Quantize model.
        Args:
            group_size: Group size for quantization.
            bits: Number of bits for quantization.
        """
        self._quantization = {
            group_size: group_size,
            bits: bits
        }

        nn.QuantizedLinear.quantize_module(self.model, group_size, bits)

    def generate(self, prompt, temp=0.0, num_tokens=256, flush=5):
        """
        Iterator for generating tokens.
        Args:
            prompt: Prompt to start generation.
            temp: Sampling temperature.
            num_tokens: Max number of tokens to generate.
            flush: Flush every `flush` tokens.
        """
        prompt = mx.array(self.tokenizer.encode(prompt))

        def generate_step():
            def sample(logits):
                if temp == 0:
                    return mx.argmax(logits, axis=-1)
                else:
                    return mx.random.categorical(logits * (1 / temp))

            logits, cache = self.model(prompt[None])
            y = sample(logits[:, -1, :])
            yield y

            while True:
                logits, cache = self.model(y[:, None], cache)
                y = sample(logits.squeeze(1))
                yield y

        tokens = []
        for token, _ in zip(generate_step(), range(num_tokens)):
            if token[0] == self.eos_id:
                break
            tokens.append(token)

            if (len(tokens) % flush) == 0:
                mx.eval(tokens)
                s = self.tokenizer.decode([t.item() for t in tokens])

                yield s

                tokens = []

        mx.eval(tokens)
        s = self.tokenizer.decode([t.item() for t in tokens])

        yield s