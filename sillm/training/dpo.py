import logging

import mlx.core as mx
import mlx.nn as nn

from sillm.llm import LLM, generate
from sillm.args import ModelArgs
from sillm.training.lora import TrainableLoRA

class TrainableDPO(TrainableLoRA):
    """
    Trainable DPO model wrapper.
    """
    @staticmethod
    def from_model(llm: LLM, **kwargs):
        """
        Convert LLM to trainable DPO LLM.
        Args:
            llm: LLM to convert.
        Returns:
            Trainable LLM.
        """
        model = TrainableDPO(llm.model, llm.tokenizer, llm.args, **kwargs)
        model._quantization = llm._quantization

        return model
    
    def __init__(self,
                 model,
                 tokenizer,
                 args: ModelArgs,
                 reference_free: bool = False,
                 loss_type: str = "sigmoid",
                 loss_beta: float = 0.1,
                 label_smoothing: float = 0.0
                 ):
        """
        Args:
            tokenizer: Tokenizer instance.
            args: Model arguments.
            loss_type: Type of loss function (sigmoid/hinge/ipo).
            loss_beta: Loss beta parameter.
        """
        super().__init__(model, tokenizer, args)

        self.reference_free = reference_free
        self.loss_type = loss_type
        self.beta = loss_beta
        self.label_smoothing = label_smoothing

        if reference_free:
            self.reference = None
        else:
            model_class = model.__class__
            self.reference = model_class(args)
            weights = self.model.parameters()
            self.reference.update(weights)
            self.reference.freeze()

        logging.info(f"Initialized DPO with reference model")

    def comparison(self,
                   prompt: str,
                   temp: float = 0.0,
                   num_tokens: int = 256
                   ):
        """
        Generate comparison between policy and reference model completions.
        Args:
            prompt: Prompt to start generation.
            num_tokens: Max number of tokens to generate.
        Returns:
            Completions.
        """
        reference_completion = ''.join([t[0] for t in generate(self.reference, self.tokenizer, prompt, temp=temp, num_tokens=num_tokens)])
        policy_completion = ''.join([t[0] for t in generate(self.model, self.tokenizer, prompt, temp=temp, num_tokens=num_tokens)])

        return reference_completion, policy_completion

    ########
    # References:
    # https://github.com/eric-mitchell/direct-preference-optimization
    # https://huggingface.co/docs/trl/main/en/dpo_trainer
    # https://github.com/huggingface/trl/blob/1bfe0b8fcb02d91d842cdc64e8810871d2d5fd91/trl/trainer/dpo_trainer.py#L840
    ########
    def loss(self,
            chosen: mx.array,
            rejected: mx.array,
            prompt_lengths: mx.array,
            chosen_lengths: mx.array,
            rejected_lengths: mx.array):
        """
        Calculate loss for inputs.
        Args:
            inputs: Input tokens.
            targets: Target tokens.
            lengths: Lengths of inputs.
        Returns:
            Loss value.
        """
        def forward(model, x):
            inputs = x[:, :-1]
            logits, _ = model(inputs)
            logits = logits.astype(mx.float32)
            targets = x[:, 1:]

            logits = nn.log_softmax(logits, axis=-1)

            return mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)

        num_toks = chosen_lengths.sum() + rejected_lengths.sum()

        # Mask prompt and padding tokens
        chosen_mask = mx.logical_and(
            mx.arange(chosen.shape[1] - 1)[None, :] < (chosen_lengths[:, None] - 1),
            mx.arange(chosen.shape[1] - 1)[None, :] > (prompt_lengths[:, None] - 1)
        )
        rejected_mask = mx.logical_and(
            mx.arange(rejected.shape[1] - 1)[None, :] < (rejected_lengths[:, None] - 1),
            mx.arange(rejected.shape[1] - 1)[None, :] > (prompt_lengths[:, None] - 1)
        )

        # Calculate log probabilities for policy model
        policy_chosen_scores = forward(self.model, chosen) * chosen_mask
        policy_rejected_scores = forward(self.model, rejected) * rejected_mask
        if self.loss_type == "ipo":
            # ipo uses average log probabilities
            policy_chosen_score = policy_chosen_scores.sum(-1) / chosen_mask.sum(-1)
            policy_rejected_score = policy_rejected_scores.sum(-1) / rejected_mask.sum(-1)
        else:
            policy_chosen_score = policy_chosen_scores.sum(-1)
            policy_rejected_score = policy_rejected_scores.sum(-1)
        policy_score = policy_chosen_score - policy_rejected_score

        # Calculate log probabilities for reference model
        if self.reference_free:
            reference_chosen_score = mx.zeros_like(policy_chosen_score)
            reference_rejected_score = mx.zeros_like(policy_rejected_score)
            reference_score = mx.zeros_like(policy_score)
        else:
            reference_chosen_scores = mx.stop_gradient(forward(self.reference, chosen)) * chosen_mask
            reference_rejected_scores = mx.stop_gradient(forward(self.reference, rejected)) * rejected_mask
            if self.loss_type == "ipo":
                # ipo uses average log probabilities
                reference_chosen_score = reference_chosen_scores.sum(-1) / chosen_mask.sum(-1)
                reference_rejected_score = reference_rejected_scores.sum(-1) / rejected_mask.sum(-1)
            else:
                reference_chosen_score = reference_chosen_scores.sum(-1)
                reference_rejected_score = reference_rejected_scores.sum(-1)
            reference_score = reference_chosen_score - reference_rejected_score
        
        ratios = policy_score - reference_score

        if self.loss_type == "sigmoid":
            # https://arxiv.org/abs/2305.18290
            losses = -nn.log_sigmoid(self.beta * ratios)
        elif self.loss_type == "hinge":
            # https://arxiv.org/abs/2309.06657
            losses = nn.relu(1 - self.beta * ratios)
        elif self.loss_type == "ipo":
            # https://arxiv.org/abs/2310.12036
            losses = (ratios - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "dpop":
            # https://arxiv.org/abs/2402.13228v1
            self.delta = 50
            penalty = mx.maximum(mx.zeros_like(policy_chosen_score), reference_chosen_score - policy_chosen_score)
            losses = -nn.log_sigmoid(self.beta * ratios - self.delta * penalty)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        loss = mx.mean(losses)
        
        print("policy_chosen_score", policy_chosen_score)
        print("policy_rejected_score", policy_rejected_score)
        print("policy_score", policy_score)
        if self.reference_free is False:
            print("reference_chosen_score", reference_chosen_score)
            print("reference_rejected_score", reference_rejected_score)
            print("reference_score", reference_score)
        print("ratios", ratios)
        print("penalty", penalty)
        print("losses", mx.mean(losses))

        return loss, num_toks