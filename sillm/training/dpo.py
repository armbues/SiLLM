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
        """
        # prompt = mx.array(self.model.tokenizer.encode("Hello World!"))
        # logits, _ = self.model.model(prompt)
        # probs = mx.log(mx.softmax(logits, axis=-1))