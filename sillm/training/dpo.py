import logging

import mlx.core as mx

from sillm.llm import LLM
from sillm.args import ModelArgs
from sillm.training.lora import TrainableLoRA
from sillm.training.dataset import Dataset

class TrainableDPO(TrainableLoRA):
    """
    Trainable DPO model wrapper.
    """
    @staticmethod
    def from_model(llm: LLM):
        """
        Convert LLM to trainable DPO LLM.
        Args:
            llm: LLM to convert.
            args: Model arguments.
        Returns:
            Trainable LLM.
        """
        model = TrainableDPO(llm.model, llm.tokenizer, llm.args)
        model._quantization = llm._quantization

        return model
    
    def __init__(self,
                 model,
                 tokenizer,
                 args: ModelArgs,
                 beta: float = 0.1
                 ):
        """
        Args:
            tokenizer: Tokenizer instance.
            args: Model arguments.
            beta: Beta parameter.
        """
        super().__init__(model, tokenizer, args)

        self.beta = beta

        model_class = model.__class__
        self.reference = model_class(args)
        weights = self.model.parameters()
        self.reference.update(weights)
        self.reference.freeze()

        logging.info(f"Initialized DPO with additional reference model")

    ########
    # References:
    # https://github.com/huggingface/trl/blob/1bfe0b8fcb02d91d842cdc64e8810871d2d5fd91/trl/trainer/dpo_trainer.py#L840
    # https://github.com/lucidrains/self-rewarding-lm-pytorch/blob/81fc3df92e3bff77b737a3428f49ff7de4dd0057/self_rewarding_lm_pytorch/dpo.py#L282
    ########
    def loss(self,
            chosen: mx.array,
            rejected: mx.array,
            lengths: mx.array):
        """
        Calculate loss for inputs.
        Args:
            inputs: Input tokens.
            targets: Target tokens.
            lengths: Lengths of inputs.
        Returns:
            Loss value.
        """
        def log_prob(model, inputs):
            logits, _ = model(inputs[:, :-1])

            # prob = mx.softmax(logits, axis=-1)
            x_shifted = logits - mx.max(logits)
            log_sum_exp = mx.log(mx.sum(mx.exp(x_shifted)))
            prob = x_shifted - log_sum_exp

            targets = inputs[0][1:]
            targets = targets.reshape(targets.shape[0], 1)

            return mx.take_along_axis(prob[0], targets, axis=-1).flatten()

        # Calculate log probabilities for reference model
        reference_chosen_logprobs = mx.stop_gradient(log_prob(self.reference, chosen))
        reference_rejected_logprobs = mx.stop_gradient(log_prob(self.reference, rejected))
        reference_chosen_logprob = mx.mean(reference_chosen_logprobs, axis=-1)
        reference_rejected_logprob = mx.mean(reference_rejected_logprobs, axis=-1)

        # Calculate log probabilities for policy model
        policy_chosen_logprobs = log_prob(self.model, chosen)
        policy_rejected_logprobs = log_prob(self.model, rejected)
        policy_chosen_logprob = mx.mean(policy_chosen_logprobs, axis=-1)
        policy_rejected_logprob = mx.mean(policy_rejected_logprobs, axis=-1)

        reference_logratios = reference_chosen_logprob - reference_rejected_logprob
        policy_logratios = policy_chosen_logprob - policy_rejected_logprob

        losses = -mx.log(mx.sigmoid(self.beta * (policy_logratios - reference_logratios)))

        return mx.mean(losses), lengths.sum()