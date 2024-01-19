import pathlib
import time
import logging

import math
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from sillm.llm import LLM
from sillm.args import ModelArgs
from sillm.dataset import Dataset

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/lora/models.py#L55
########
class LoRALinear(nn.Module):
    """
    Linear layer with LoRA weights.
    """
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8):
        """
        Convert linear layer to LoRA linear layer.
        Args:
            linear: Linear layer to convert.
            rank: Rank to use for LoRA.
        Returns:
            LoRA linear layer.
        """
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
        """
        Args:
            input_dims: Input dimensions.
            output_dims: Output dimensions.
            lora_rank: Rank to use for LoRA.
            scale: Scale to use for LoRA.
            bias: Whether to use bias.
        """
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

    @property
    def lora_size(self):
        """
        Returns:
            Number of LoRA parameters.
        """
        return self.lora_a.size + self.lora_b.size
    
    def merge(self):
        """
        Merge LoRA weights into linear weights.
        Returns:
            Linear layer with merged weights.
        """
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
        """
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        dtype = self.linear.weight.dtype
        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype

        y = self.linear(x.astype(dtype))
        z = (x @ self.lora_a) @ self.lora_b

        return y + self.scale * z

class TrainableLLM(LLM):
    """
    Trainable LLM model wrapper.
    """
    def __init__(self,
                 tokenizer,
                 args: ModelArgs
                 ):
        """
        Args:
            tokenizer: Tokenizer instance.
            args: Model arguments.
        """
        super().__init__(tokenizer, args)

        self._lora = None

    def init_lora(self,
                  num_layers: int = -1,
                  target_modules: list = ["attention.wq", "attention.wv"],
                  rank: int = 8):
        """
        Initialize LoRA for model.
        Args:
            num_layers: Number of layers to apply LoRA to.
            target_modules: Modules to apply LoRA to.
            rank: Rank to use for LoRA.
        """
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

            trainable_params = 0
            for layer in self.model.layers[-num_layers:]:
                for target in target_modules:
                    sub, mod = target.split(".")
                    layer[sub][mod] = LoRALinear.from_linear(layer[sub][mod], rank=rank)
                    trainable_params += layer[sub][mod].lora_size

            logging.info(f"Initialized LoRA with rank {rank} for {num_layers} layers")
            logging.debug(f"LoRA target modules: {', '.join(target_modules)}")
            logging.debug(f"LoRA trainable parameters: {trainable_params/ 10**6:.2f}M")

    def merge_and_unload_lora(self):
        """
        Merge LoRA layers back into model.
        """
        if self._lora is not None:
            num_layers = self._lora["num_layers"]
            target_modules = self._lora["target_modules"]

            for layer in self.model.layers[-num_layers:]:
                for target in target_modules:
                    sub, mod = target.split(".")

                    if isinstance(layer[sub][mod], LoRALinear):
                        layer[sub][mod] = layer[sub][mod].merge()

            logging.info(f"Merged LoRA layers back into model")

        self._lora = None

    def save_adapters(self, adapter_path):
        """
        Save adapter weights.
        Args:
            adapter_path: Path to save adapter weights to.
        """
        assert self._lora is not None

        state = dict(tree_flatten(self.model.trainable_parameters()))
        mx.savez(adapter_path, **state)

        logging.info(f"Saved adapter weights to {adapter_path}")

    ########
    # Based on mlx-examples:
    # https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/lora/lora.py#L166
    ########
    def iterate_batches(self,
                        dataset: Dataset,
                        batch_size: int,
                        train: bool = False):
        """
        Iterate over batches.
        Args:
            dataset: Dataset to iterate over.
            batch_size: Batch size.
            train: Whether to train.
        """
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
    def evaluate(self,
                 dataset: Dataset,
                 batch_size: int,
                 num_batches: int
                 ):
        """
        Evaluate model on dataset.
        Args:
            dataset: Dataset to evaluate on.
            batch_size: Batch size.
            num_batches: Number of batches to evaluate.
        Returns:
            Average loss.
        """
        all_losses = []
        num_tokens = 0
        for _, batch in zip(
            range(num_batches),
            self.iterate_batches(dataset, batch_size),
        ):
            losses, toks = self.model.loss(*batch)
            all_losses.append((losses * toks).item())
            num_tokens += toks.item()

        return np.sum(all_losses) / num_tokens

    ########
    # Based on mlx-examples:
    # https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/lora/lora.py#L212
    ########
    def train(self, 
              dataset_training: Dataset,
              dataset_validation: Dataset,
              batch_size: int = 4,
              learning_rate: float = 1e-5,
              iterations: int = 1000,
              report_steps: int = 10,
              eval_steps: int = 100,
              validation_batches: int = 25):
        """
        Train model.
        Args:
            dataset_training: Training dataset.
            dataset_validation: Validation dataset.
            batch_size: Batch size.
            learning_rate: Learning rate.
            iterations: Number of iterations.
            report_steps: Report every `report_steps` iterations.
            eval_steps: Evaluate every `eval_steps` iterations.
            validation_batches: Number of batches to evaluate on.
        """
        logging.info(f"Training the model for {iterations} iterations with batch size {batch_size} and learning rate {learning_rate}")
        self.model._training = True
        # TODO switch off training
        
        optimizer = optim.Adam(learning_rate=learning_rate)

        # Create value and grad function for loss
        loss_value_and_grad = nn.value_and_grad(self.model, self.model.loss)

        losses = []
        num_tokens = 0

        # Main training loop
        start = time.perf_counter()
        for i, batch in zip(
            range(iterations),
            self.iterate_batches(dataset_training, batch_size, train=True),
        ):
            # Forward and backward pass
            (lvalue, toks), grad = loss_value_and_grad(*batch)

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

                logging.info(f"Iter {i + 1}: Train loss {train_loss:.3f} It/sec {report_steps / (stop - start):.3f} Tokens/sec {float(num_tokens) / (stop - start):.3f}")
                
                losses = []
                num_tokens = 0
                start = time.perf_counter()

            # Report validation loss if needed
            if i == 0 or (i + 1) % eval_steps == 0:
                stop = time.perf_counter()
                val_loss = self.evaluate(dataset_validation, batch_size, validation_batches)
                start = time.perf_counter()

                logging.info(f"Iter {i + 1}: Val loss {val_loss:.3f} Val took {(start - stop):.3f}s")

    def load_adapters(self, adapter_path: str):
        """
        Load adapter weights.
        Args:
            adapter_path: Path to adapter weights.
        """
        assert pathlib.Path(adapter_path).exists(), adapter_path

        self.model.load_weights(adapter_path)

        logging.info(f"Loaded adapter weights from {adapter_path}")