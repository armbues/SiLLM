import pathlib
import json
import time

import math
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

import sillm.llm as llm
import sillm.model as model

class Dataset:
    def __init__(self, tokenizer, dataset_path, key="text", max_length=4096):
        self._key = key
        self._data = []

        if pathlib.Path(dataset_path).exists():
            with open(dataset_path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    text = entry[key]
                    tokens = tokenizer.encode(text, eos=True)

                    if len(tokens) < max_length:
                        self._data.append(tokens)

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)
    
    @staticmethod
    def load(tokenizer, dataset_path, key="text", max_length=4096):
        names = ("train", "valid", "test")

        return (Dataset(tokenizer, pathlib.Path(dataset_path) / f"{n}.jsonl", key, max_length) for n in names)

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/lora/models.py#L55
########
class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8):
        output_dims, input_dims = linear.weight.shape

        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits

        lora_lin = LoRALinear(input_dims, output_dims, rank)
        lora_lin.linear = linear

        return lora_lin

    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 lora_rank: int = 8,
                 scale : float = 2.0,
                 bias: bool = False):
        super().__init__()

        # Initialize linear layer weights
        self.scale = scale
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Initialize LoRA weights
        bound = 1 / math.sqrt(input_dims)
        input_shape = (input_dims, lora_rank)
        output_shape = (lora_rank, output_dims)

        self.lora_a = mx.random.uniform(low=-bound, high=bound, shape=input_shape)
        self.lora_b = mx.zeros(shape=output_shape)

    def merge(self):
        linear = self.linear
        weight = linear.weight
        dtype = linear.weight.dtype

        quantized = isinstance(linear, nn.QuantizedLinear)
        if quantized:
            dtype = mx.float16
            group_size = linear.group_size
            bits = linear.bits
            weight = mx.dequantize(weight, linear.scales, linear.biases, group_size, bits)

        # Merge LoRA weights into linear weights
        update = (self.lora_a @ self.lora_b).transpose()
        weight = (weight + (self.scale * update)).astype(dtype)

        if quantized:
            output_dims, input_dims = weight.shape
            bias = "bias" in linear
            linear = nn.Linear(input_dims, output_dims, bias=bias)

            return nn.QuantizedLinear.from_linear(linear, group_size, bits)
        else:
            linear.weight = weight

            return linear

    def __call__(self, x):
        dtype = self.linear.weight.dtype
        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype

        y = self.linear(x.astype(dtype))
        z = (x @ self.lora_a) @ self.lora_b

        return y + self.scale * z

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/047d4650c4f63d55e5bfbaf8f589c1679cbdd971/lora/lora.py#L151
########
def loss(model, inputs, targets, lengths):
    # Run model on inputs
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks

class TrainableLLM(llm.LLM):
    def __init__(self, tokenizer, args: model.ModelArgs):
        super().__init__(tokenizer, args)

        self._lora = None

    def init_lora(self,
                  num_layers: int = -1,
                  target_modules: list = ["attention.wq", "attention.wv"],
                  rank: int = 8):
        if self._lora is None:
            if num_layers < 0:
                num_layers = len(self.model.layers)

            self._lora = {
                "num_layers": num_layers,
                "target_modules": target_modules,
                "rank": rank
            }
            
            # Freeze all existing parameters
            self.model.freeze()

            for layer in self.model.layers[-num_layers:]:
                for target in target_modules:
                    sub, mod = target.split(".")
                    layer[sub][mod] = LoRALinear.from_linear(layer[sub][mod], rank=rank)

    def merge_and_unload_lora(self):
        if self._lora is not None:
            num_layers = self._lora["num_layers"]
            target_modules = self._lora["target_modules"]

            for layer in self.model.layers[-num_layers:]:
                for target in target_modules:
                    sub, mod = target.split(".")

                    if isinstance(layer[sub][mod], LoRALinear):
                        layer[sub][mod] = layer[sub][mod].merge()

        self._lora = None

    def save_adapters(self, adapter_path):
        assert self._lora is not None

        state = dict(tree_flatten(self.model.trainable_parameters()))
        mx.savez(adapter_path, **state)

    ########
    # Based on mlx-examples:
    # https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/lora/lora.py#L166
    ########
    def iterate_batches(self, dataset, batch_size, train=False):
        # Shuffle indices
        while True:
            indices = np.arange(len(dataset))
            if train:
                indices = np.random.permutation(indices)

            # Collect batches from dataset
            for i in range(0, len(indices) - batch_size + 1, batch_size):
                batch = [dataset[i+j] for j in range(batch_size)]
                lengths = [len(x) for x in batch]

                # Pad to the max length
                batch_arr = np.zeros((batch_size, max(lengths)), np.int32)
                for j in range(batch_size):
                    batch_arr[j, : lengths[j]] = batch[j]
                batch = mx.array(batch_arr)

                yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

            if not train:
                break

    ########
    # Based on mlx-examples:
    # https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/lora/lora.py#L198
    ########
    def evaluate(self, dataset, loss, batch_size, num_batches):
        all_losses = []
        num_tokens = 0
        for _, batch in zip(
            range(num_batches),
            self.iterate_batches(dataset, batch_size),
        ):
            losses, toks = loss(self.model, *batch)
            all_losses.append((losses * toks).item())
            num_tokens += toks.item()

        return np.sum(all_losses) / num_tokens

    ########
    # Based on mlx-examples:
    # https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/lora/lora.py#L212
    ########
    def train(self, 
              dataset_training,
              dataset_validation,
              batch_size: int = 4,
              learning_rate: float = 1e-5,
              iterations: int = 1000,
              report_steps: int = 10,
              eval_steps: int = 100,
              validation_batches: int = 25):
        optimizer = optim.Adam(learning_rate=learning_rate)

        # Create value and grad function for loss
        loss_value_and_grad = nn.value_and_grad(self.model, loss)

        losses = []
        num_tokens = 0

        # Main training loop
        start = time.perf_counter()
        for i, batch in zip(
            range(iterations),
            self.iterate_batches(dataset_training, batch_size, train=True),
        ):
            # Forward and backward pass
            (lvalue, toks), grad = loss_value_and_grad(self.model, *batch)

            # Model update
            optimizer.update(self.model, grad)
            mx.eval(self.model.parameters(), optimizer.state, lvalue)

            # Record loss
            losses.append(lvalue.item())
            num_tokens += toks.item()

            # Report training loss if needed
            if (i + 1) % report_steps == 0:
                train_loss = np.mean(losses)

                stop = time.perf_counter()
                print(
                    f"Iter {i + 1}: Train loss {train_loss:.3f}, "
                    f"It/sec {report_steps / (stop - start):.3f}, "
                    f"Tokens/sec {float(num_tokens) / (stop - start):.3f}"
                )
                losses = []
                num_tokens = 0
                start = time.perf_counter()

            # Report validation loss if needed
            if i == 0 or (i + 1) % eval_steps == 0:
                stop = time.perf_counter()
                val_loss = self.evaluate(dataset_validation, loss, batch_size, validation_batches)
                print(
                    f"Iter {i + 1}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {(time.perf_counter() - stop):.3f}s"
                )

                start = time.perf_counter()

    def load_adapter(self, adapter_path: str):
        assert pathlib.Path(adapter_path).exists(), adapter_path

        self.model.load_weights(adapter_path)