import logging

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from sillm.core.llm import LLM
from sillm.models.args import ModelArgs
from sillm.training.lora import TrainableLoRA

logger = logging.getLogger("sillm")

########
# References:
# Geoffrey Hinton, Oriol Vinyals and Jeff Dean. Distilling the Knowledge in a Neural Network. https://arxiv.org/abs/1503.02531
########
class Distillation(TrainableLoRA):
    @staticmethod
    def from_model(target_llm: LLM, draft_llm: LLM, **kwargs):
        """
        Convert LLM to trainable distillation LLM.
        Args:
            llm: LLM to convert.
        Returns:
            Trainable distillation LLM.
        """
        model = Distillation(target_llm.model, draft_llm.model, target_llm.tokenizer, draft_llm.tokenizer, target_llm.args, **kwargs)
        model._quantization = target_llm._quantization

        return model

    def __init__(self,
                 target_model,
                 draft_model,
                 target_tokenizer,
                 draft_tokenizer,
                 args: ModelArgs,
                 loss_alpha: float = 0.1
                 ):
        """
        Args:
            target_model: Target model instance.
            draft_model: Draft model instance.
            target_tokenizer: Target tokenizer instance.
            draft_tokenizer: Draft tokenizer instance.
            args: Model arguments.
            loss_alpha: Alpha parameter for distillation loss.
            loss_beta: Beta parameter for distillation loss.
        """
        if draft_tokenizer.vocab != target_tokenizer.vocab:
            raise ValueError("Target and draft tokenizers must have the same vocabulary")
        self.vocab_size = len(target_tokenizer.vocab)

        super().__init__(target_model, target_tokenizer, args)

        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer

        self.alpha = loss_alpha
        self.beta = 1.0 - loss_alpha

        logger.info(f"Initialized model distillation with alpha {loss_alpha}")
        
    def loss(self,
             inputs: mx.array,
             targets: mx.array,
             loss_masks: mx.array
             ):
        num_tokens = loss_masks.sum()

        # Calculate student loss
        student_logits = self.model(inputs)[:, :, :self.vocab_size]
        student_loss = nn.losses.cross_entropy(student_logits, targets) * loss_masks
        student_loss = student_loss.sum() / num_tokens

        # Calculate distillation loss
        teacher_logits = self.draft_model(inputs)[:, :, :self.vocab_size]
        teacher_probs = nn.softmax(teacher_logits)
        distill_loss = nn.losses.cross_entropy(student_logits, teacher_probs) * loss_masks
        distill_loss = distill_loss.sum() / num_tokens

        loss_value = self.alpha * student_loss + self.beta * distill_loss
        reward = mx.stack([student_loss, distill_loss])
        
        return loss_value, reward, num_tokens