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
        Convert LLM to trainable LLM.
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
                 args: ModelArgs
                 ):
        """
        Args:
            tokenizer: Tokenizer instance.
            args: Model arguments.
        """
        super().__init__(model, tokenizer, args)

    ########
    # References:
    # https://github.com/lucidrains/self-rewarding-lm-pytorch/blob/ec8b9112d4ced084ae7cacfe776e1ec01fa1f950/self_rewarding_lm_pytorch/dpo.py#L282
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
        def log_prob(prompt):
            logits, _ = self.model(prompt)
            prob = mx.softmax(logits, axis=-1)

            return mx.log(prob)
        
        # Calculate log probabilities for reference model
        self.model.toggle_lora(False)
        reference_chosen_logprobs = mx.stop_gradient(log_prob(chosen))
        reference_rejected_logprobs = mx.stop_gradient(log_prob(rejected))
        self.model.toggle_lora(True)

        # Calculate log probabilities for policy model
        policy_chosen_logprobs = log_prob(chosen)
        policy_rejected_logprobs = log_prob(rejected)