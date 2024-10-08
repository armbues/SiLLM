import math

import mlx.core as mx
import mlx.nn as nn

from sillm.models.args import ModelArgs

class BaseModel(nn.Module):
    """
    Base class for LLM models.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
    
    def __call__(self,
                 inputs: mx.array,
                 cache = None
                 ):
        raise NotImplementedError("Class model.Model is used for inheritance only")
    
    ########
    # Based on mlx-examples:
    # https://github.com/ml-explore/mlx-examples/blob/047d4650c4f63d55e5bfbaf8f589c1679cbdd971/lora/lora.py#L151
    ########
    def loss(self,
             inputs: mx.array,
             targets: mx.array,
             loss_masks: mx.array
             ):
        """
        Calculate cross-entropy loss.
        Args:
            inputs: Input tokens.
            targets: Target tokens.
            lengths: Lengths of inputs.
        Returns:
            Cross-entropy loss.
        """
        # Run model on inputs
        logits = self.__call__(inputs)
        logits = logits.astype(mx.float32)

        # Calculate the loss
        cross_entropy_loss = nn.losses.cross_entropy(logits, targets) * loss_masks
        num_tokens = loss_masks.sum()
        loss_value = cross_entropy_loss.sum() / num_tokens

        return loss_value, None, num_tokens

    @staticmethod    
    def create_additive_causal_mask(N: int, offset: int = 0):
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        mask = linds[:, None] < rinds[None]
        
        return mask * -1e9