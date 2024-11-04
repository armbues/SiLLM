import os
import time
import logging

from functools import partial

import tqdm
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from sillm.core.llm import LLM
from sillm.training.dataset import Dataset

logger = logging.getLogger("sillm")

class TrainableLLM(LLM):
    """
    Trainable LoRA model wrapper.
    """
    @staticmethod
    def from_model(llm: LLM):
        """
        Convert LLM to trainable LLM.
        Args:
            llm: LLM to convert.
        Returns:
            Trainable LLM.
        """
        return TrainableLLM(llm.model, llm.tokenizer, llm.args)
    
    def __init__(self, llm: LLM):
        """
        Args:
            tokenizer: Tokenizer instance.
            args: Model arguments.
        """
        self.model = llm.model
        self.tokenizer = llm.tokenizer
        self.args = llm.args

        self._quantization = llm._quantization
        self._mapping = llm._mapping

    def loss(self, *args, **kwargs):
        """
        Default loss function from model.
        """
        return self.model.loss(*args, **kwargs)
    
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
        losses = []
        for _, batch in zip(
            range(num_batches),
            dataset.iterate_batches(batch_size),
        ):
            loss_value, _, _ = self.loss(*batch)
            losses.append(loss_value.item())

        return np.mean(losses)
    
    ########
    # Based on mlx-examples:
    # https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/lora/lora.py#L212
    ########
    def train(self, 
              dataset_training: Dataset,
              dataset_validation: Dataset,
              batch_size: int = 4,
              optimizer_type: str = "adam",
              learning_rate: float = 1e-5,
              learning_decay: float = 0.0,
              learning_warmup: int = 0,
              compiled_step: bool = True,
              gradient_checkpointing: bool = False,
              gradient_accumulation_steps: int = 1,
              gradient_max_norm: float = 1.0,
              epochs: int = 1,
              iterations: int = 0,
              report_steps: int = 10,
              report_callback: callable = None,
              eval_steps: int = 100,
              eval_callback: callable = None,
              validation_samples: int = 40
              ):
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
            validation_samples: Number of validation samples.
            debug: Whether to enable debug mode.
        """
        # Calculate number of iterations
        if iterations == 0:
            iterations = len(dataset_training) // batch_size
        
        # Calculate number of validation batches
        validation_batches = validation_samples // batch_size
        
        logger.info(f"Training the model for {epochs} epochs of {iterations} batch iterations with batch size {batch_size}")
        logger.debug(f"Training learning rate: {learning_rate}")

        # Get system memory
        system_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

        # Initialize scheduler
        if learning_decay > 0.0:
            scheduler = optim.step_decay(learning_rate, 1-learning_decay, 1)

            logger.debug(f"Training learning decay: {learning_decay}")
        else:
            scheduler = optim.linear_schedule(learning_rate, learning_rate, 1)
        if learning_warmup > 0:
            warmup = optim.linear_schedule(learning_rate * 0.1, learning_rate, learning_warmup)
            scheduler = optim.join_schedules([warmup, scheduler], [learning_warmup])

            logger.debug(f"Training learning warmup steps: {learning_warmup}")

        # Initialize optimizer
        optimizer_type = optimizer_type.lower()
        if optimizer_type == "adam":
            optimizer = optim.Adam(learning_rate=scheduler)
        elif optimizer_type == "adamw":
            optimizer = optim.AdamW(learning_rate=scheduler, weight_decay=0.0)
        elif optimizer_type == "adafactor":
            optimizer = optim.Adafactor(learning_rate=scheduler, weight_decay=0.0)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Initialize gradient accumulation
        if gradient_accumulation_steps > 1:
            accum_grad = None
            accum_scale = 1 / gradient_accumulation_steps

            if gradient_checkpointing:
                gradient_checkpointing = False
                logger.warning(f"Gradient accumulation requires disabling gradient checkpointing")
            if compiled_step:
                compiled_step = False
                logger.warning(f"Gradient accumulation requires disabling compiled step function")

            logger.info(f"Enabled gradient accumulation with {gradient_accumulation_steps} steps")

        # Initialize gradient checkpointing
        if gradient_checkpointing:
            if not compiled_step:
                logger.warning(f"Gradient checkpointing requires compiled step function")
                compiled_step = True
            
            for layer in self.model.layers:
                layer.forward = nn.utils.checkpoint(layer, layer.forward)

            logger.info(f"Enabled gradient checkpointing")

        # Create value and gradient function for loss
        loss_value_and_grad = nn.value_and_grad(self.model, self.loss)

        # Initialize compiled step function
        if compiled_step:
            state = [self.model.state, optimizer.state]

            # Step function for forward and backward pass
            @partial(mx.compile, inputs=state, outputs=state)
            def step(batch):
                (loss_value, reward, num_tokens), grad = loss_value_and_grad(*batch)
                optimizer.update(self.model, grad)

                return loss_value, reward, num_tokens

        # Initialize variables for training loop
        losses = []
        rewards = None
        intv_tokens = 0

        # Main training loop
        start = time.perf_counter()
        pbar_epochs = tqdm.tqdm(range(epochs), desc="Epoch")
        for epoch in pbar_epochs:
            pbar_iterations = tqdm.tqdm(range(iterations), desc="Iter.", leave=False)
            for iter in pbar_iterations:
                n = epoch * iterations + iter
                batch = next(dataset_training.iterate_batches(batch_size, train=True))

                if compiled_step:
                    loss_value, reward, num_tokens = step(batch)
                else:
                    (loss_value, reward, num_tokens), grad = loss_value_and_grad(*batch)

                    if gradient_accumulation_steps > 1:
                        # Accumulate gradients
                        if accum_grad is None:
                            accum_grad = grad
                        else:
                            accum_grad = tree_map(mx.add, grad, accum_grad)
                        mx.eval(accum_grad)
                        del grad

                        # Update model with accumulated gradients
                        if (n + 1) % gradient_accumulation_steps == 0:
                            accum_grad, _ = optim.clip_grad_norm(accum_grad, max_norm=gradient_max_norm)
                            accum_grad = tree_map(lambda g: g * accum_scale, accum_grad)
                            optimizer.update(self.model, accum_grad)
                            mx.eval(optimizer.state)
                            accum_grad = None
                    else:
                        # Update model with gradients
                        grad, _ = optim.clip_grad_norm(grad, max_norm=gradient_max_norm)
                        optimizer.update(self.model, grad)
                        mx.eval(optimizer.state)

                # Evaluate loss & reward
                mx.eval(loss_value, reward, num_tokens)

                # Record loss and number of tokens
                losses.append(loss_value.item())
                intv_tokens += num_tokens.item()

                # Record rewards
                if reward is not None:
                    if rewards is None:
                        rewards = reward
                    else:
                        rewards = np.vstack([rewards, reward])

                # Report training loss if needed
                if (n + 1) % report_steps == 0:
                    train_loss = np.mean(losses)
                    stop = time.perf_counter()

                    # Print training loss and timings
                    pbar_epochs.write(f"#{n + 1}:\tTraining loss    {train_loss:.3f}\t{float(intv_tokens) / (stop - start):.3f} tok/sec (learning rate: {optimizer.learning_rate.item():.3e})")
                    if rewards is not None:
                        pbar_epochs.write(f"#{n + 1}:\tTraining reward  {str(np.mean(rewards, axis=0))}")
                        rewards = None

                    # Verbose logging using tqdm output
                    if logger.level <= logging.INFO:
                        # Print memory usage
                        peak_memory = mx.metal.get_peak_memory()
                        memory_usage = peak_memory / system_memory
                        pbar_epochs.write(f"#{n + 1}:\tPeak memory      {(peak_memory // (1024 ** 2)):,} MB ({memory_usage:.2%} of system memory)")
                        mx.metal.reset_peak_memory()

                    pbar_epochs.refresh()

                    if report_callback is not None:
                        report_callback(n + 1, train_loss)
                    
                    losses = []
                    intv_tokens = 0
                    start = time.perf_counter()

                # Report validation loss if needed
                if n == 0 or (n + 1) % eval_steps == 0:
                    # Print validation loss and timings
                    stop = time.perf_counter()
                    val_loss = self.evaluate(dataset_validation, batch_size, validation_batches)
                    start = time.perf_counter()
                    pbar_epochs.write(f"#{n + 1}:\tValidation loss  {val_loss:.3f}\t{(start - stop):.3f} sec")

                    # Eval callback
                    if eval_callback is not None:
                        msg = eval_callback(n + 1, val_loss)
                        if msg:
                            pbar_epochs.write(f"#{n + 1}:\t" + msg)

                    start = time.perf_counter()