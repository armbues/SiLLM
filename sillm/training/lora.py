import pathlib
import time
import logging
import math

import tqdm
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from sillm.llm import LLM
from sillm.args import ModelArgs
from sillm.training.dataset import Dataset

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/lora/models.py#L55
# https://github.com/ml-explore/mlx-examples/blob/854ad8747a9c703773adf8866602b114f68aa54a/llms/mlx_lm/tuner/lora.py#L7
########
class LoRALinear(nn.Module):
    """
    Linear layer with LoRA weights.
    """
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.05,
        scale : float = 10.0
        ):
        """
        Convert linear layer to LoRA linear layer.
        Args:
            linear: Linear layer to convert.
            rank: Rank to use for LoRA.
            alpha: Alpha to use for LoRA.
            dropout: Dropout to use for LoRA.
            scale: Scale to use for LoRA.
        Returns:
            LoRA linear layer.
        """
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        bias = "bias" in linear

        lora_lin = LoRALinear(input_dims, output_dims, rank, alpha, dropout, scale, bias)
        lora_lin.linear = linear
        lora_lin.name = linear.name

        return lora_lin

    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 rank: int = 8,
                 alpha: float = 16,
                 dropout: float = 0.0,
                 scale : float = 10.0,
                 bias: bool = False):
        """
        Args:
            input_dims: Input dimensions.
            output_dims: Output dimensions.
            rank: Rank to use for LoRA.
            alpha: Alpha to use for LoRA.
            dropout: Dropout to use for LoRA.
            scale: Scale to use for LoRA.
            bias: Whether to use bias.
        """
        super().__init__()

        # Initialize linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Initialize LoRA dropout
        self.lora_dropout = nn.Dropout(p=dropout)

        # Initialize LoRA weights
        self.scale = scale * (alpha / rank)
        bound = 1 / math.sqrt(input_dims)
        input_shape = (input_dims, rank)
        output_shape = (rank, output_dims)

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
        z = (self.lora_dropout(x) @ self.lora_a) @ self.lora_b

        return y + self.scale * z

class TrainableLoRA(LLM):
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

    @staticmethod
    def from_model(llm: LLM):
        """
        Convert LLM to trainable LLM.
        Args:
            llm: LLM to convert.
            args: Model arguments.
        Returns:
            Trainable LLM.
        """
        return TrainableLoRA(llm.tokenizer, llm.args)

    def init_lora(self,
                  num_layers: int = -1,
                  target_modules: list = ["attention.wq", "attention.wv"],
                  rank: int = 8,
                  alpha: float = 16,
                  dropout: float = 0.05,
                  scale : float = 10.0):
        """
        Initialize LoRA for model.
        Args:
            num_layers: Number of layers to apply LoRA to.
            target_modules: Modules to apply LoRA to.
            rank: Rank to use for LoRA.
            alpha: Alpha to use for LoRA.
            dropout: Dropout to use for LoRA.
            scale: Scale to use for LoRA.
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

            self._lora["modules"] = {}
            for layer in self.model.layers[-num_layers:]:
                for target in target_modules:
                    sub, mod = target.split(".")
                    module = LoRALinear.from_linear(layer[sub][mod], rank=rank, alpha=alpha, dropout=dropout, scale=scale)
                    layer[sub][mod] = module
                    self._lora["modules"][module.name] = module

                # Add LoRA for MoE gates
                if hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "gate"):
                    module = LoRALinear.from_linear(layer.feed_forward.gate, rank=rank, alpha=alpha, dropout=dropout, scale=scale)
                    layer.feed_forward.gate = module
                    self._lora["modules"][module.name] = module

            logging.info(f"Initialized LoRA with rank {rank} for {num_layers} layers")
            logging.debug(f"LoRA target modules: {', '.join(target_modules)}")

            trainable_params = 0
            for module in self._lora["modules"].values():
                trainable_params += module.lora_size
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

    def save_adapters(self,
                      adapter_path: str,
                      ):
        """
        Save adapter weights.
        Args:
            adapter_path: Path to save adapter weights to.
        """
        assert self._lora is not None

        state = dict(tree_flatten(self.model.trainable_parameters()))
        mx.savez(adapter_path, **state)

        logging.info(f"Saved adapter weights to {adapter_path}")

    def save_checkpoint(self,
                        checkpoint_path: str,
                        steps: int
                        ):
        """
        Save model checkpoint.
        Args:
            checkpoint_path: Director to save checkpoints to.
            steps: Number of steps.
        """
        assert self._lora is not None

        checkpoint_path = pathlib.Path(checkpoint_path)
        adapter_path = checkpoint_path / f"ckpt-{steps}.safetensors"

        state = dict(tree_flatten(self.model.trainable_parameters()))
        mx.savez(adapter_path, **state)

        logging.info(f"Saved adapter checkpoint to {checkpoint_path}")

    def load_adapters(self,
                      adapter_path: str
                      ):
        """
        Load adapter weights.
        Args:
            adapter_path: Path to adapter weights.
        """
        assert pathlib.Path(adapter_path).exists(), adapter_path

        self.model.load_weights(adapter_path)

        logging.info(f"Loaded adapter weights from {adapter_path}")

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
            dataset.iterate_batches(batch_size),
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
              epochs: int = 1,
              iterations: int = 0,
              report_steps: int = 10,
              eval_steps: int = 100,
              eval_callback: callable = None,
              validation_batches: int = 25):
        """
        Train model.
        Args:
            dataset_training: Training dataset.
            dataset_validation: Validation dataset.
            batch_size: Batch size.
            learning_rate: Learning rate.
            epochs: Number of epochs.
            iterations: Number of iterations.
            report_steps: Report every `report_steps` iterations.
            eval_steps: Evaluate every `eval_steps` iterations.
            eval_callback: Callback after eval.
            validation_batches: Number of batches to evaluate on.
        """
        # Calculate number of iterations
        if iterations == 0:
            iterations = len(dataset_training) // batch_size
        
        logging.info(f"Training the model for {epochs} epochs of {iterations} batch iterations")
        logging.debug(f"Training batch size: {batch_size}")
        logging.debug(f"Training learning rate: {learning_rate}")
        
        optimizer = optim.Adam(learning_rate=learning_rate)

        # Create value and grad function for loss
        loss_value_and_grad = nn.value_and_grad(self.model, self.model.loss)

        losses = []
        num_tokens = 0

        # Main training loop
        start = time.perf_counter()
        pbar_epochs = tqdm.tqdm(range(epochs), desc="Epoch")
        pbar_iterations = tqdm.tqdm(range(iterations), desc="Iter.")
        for _ in pbar_epochs:
            for i in pbar_iterations:
                batch = next(dataset_training.iterate_batches(batch_size, train=True))

                # Forward and backward pass
                (loss_value, toks), grad = loss_value_and_grad(*batch)

                # Model update
                optimizer.update(self.model, grad)
                mx.eval(self.model.parameters(), optimizer.state, loss_value)

                # Record loss
                losses.append(loss_value.item())
                num_tokens += toks.item()

                # Report training loss if needed
                if (i + 1) % report_steps == 0:
                    train_loss = np.mean(losses)
                    stop = time.perf_counter()

                    pbar_epochs.write(f"#{i + 1}:\tTraining loss   {train_loss:.3f}\t{float(num_tokens) / (stop - start):.3f} tok/sec")
                    
                    losses = []
                    num_tokens = 0
                    start = time.perf_counter()

                # Report validation loss if needed
                if i == 0 or (i + 1) % eval_steps == 0:
                    pbar_epochs.write(f"#{i + 1}:\tEvaluating loss")

                    stop = time.perf_counter()
                    val_loss = self.evaluate(dataset_validation, batch_size, validation_batches)
                    start = time.perf_counter()

                    pbar_epochs.write(f"#{i + 1}:\tValidation loss {val_loss:.3f}\t{(start - stop):.3f} sec")

                    # Eval callback
                    eval_callback(i, val_loss)