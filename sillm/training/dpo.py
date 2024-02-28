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
    ########
    def loss(self,
            chosen: mx.array,
            rejected: mx.array,
            chosen_masks: mx.array,
            rejected_masks: mx.array,
            ):
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
            targets = x[:, 1:]
            
            logits, _ = model(inputs)
            logits = logits.astype(mx.float32)

            return -nn.losses.cross_entropy(logits, targets)

        num_chosen_tokens = chosen_masks.sum(-1)
        num_rejected_tokens = rejected_masks.sum(-1)

        # Calculate log probabilities for policy model
        policy_chosen_scores = forward(self.model, chosen) * chosen_masks
        policy_rejected_scores = forward(self.model, rejected) * rejected_masks
        if self.loss_type == "ipo":
            # ipo uses average log probabilities
            policy_chosen_score = policy_chosen_scores.sum(-1) / num_chosen_tokens
            policy_rejected_score = policy_rejected_scores.sum(-1) / num_rejected_tokens
        else:
            policy_chosen_score = policy_chosen_scores.sum(-1)
            policy_rejected_score = policy_rejected_scores.sum(-1)

        # Calculate log probabilities for reference model
        if self.reference_free:
            reference_chosen_score = mx.zeros_like(policy_chosen_score)
            reference_rejected_score = mx.zeros_like(policy_rejected_score)
        else:
            reference_chosen_scores = mx.stop_gradient(forward(self.reference, chosen)) * chosen_masks
            reference_rejected_scores = mx.stop_gradient(forward(self.reference, rejected)) * rejected_masks
            if self.loss_type == "ipo":
                # ipo uses average log probabilities
                reference_chosen_score = reference_chosen_scores.sum(-1) / num_chosen_tokens
                reference_rejected_score = reference_rejected_scores.sum(-1) / num_rejected_tokens
            else:
                reference_chosen_score = reference_chosen_scores.sum(-1)
                reference_rejected_score = reference_rejected_scores.sum(-1)
        
        logits = (policy_chosen_score - policy_rejected_score) - (reference_chosen_score - reference_rejected_score)

        if self.loss_type == "sigmoid":
            # https://arxiv.org/abs/2305.18290
            losses = -nn.log_sigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            # https://arxiv.org/abs/2309.06657
            losses = nn.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # https://arxiv.org/abs/2310.12036
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "dpop":
            # https://arxiv.org/abs/2402.13228v1
            self.delta = 50
            penalty = mx.maximum(mx.zeros_like(policy_chosen_score), reference_chosen_score - policy_chosen_score)
            losses = -(nn.log_sigmoid(self.beta * logits) - self.delta * penalty)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        loss = mx.mean(losses)
        num_tokens = (num_chosen_tokens + num_rejected_tokens).sum()

        chosen_reward = self.beta * mx.mean(policy_chosen_score - reference_chosen_score)
        rejected_reward = self.beta * mx.mean(policy_rejected_score - reference_rejected_score)
        reward = mx.stack([chosen_reward, rejected_reward])
        
        # print("policy_chosen_score", policy_chosen_score)
        # print("policy_rejected_score", policy_rejected_score)
        # if self.reference_free is False:
        #     print("reference_chosen_score", reference_chosen_score)
        #     print("reference_rejected_score", reference_rejected_score)
        # print("chosen_rewards", policy_chosen_score - reference_chosen_score)
        # print("rejected_rewards", policy_rejected_score - reference_rejected_score)
        # print("logits", logits)
        # if self.loss_type == "dpop":
        #     print("penalty", penalty)
        # print("losses", losses)
        # print("loss", loss)

        return loss, reward, num_tokens