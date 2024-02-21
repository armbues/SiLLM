import mlx.core as mx
import mlx.nn as nn

from sillm.model import BaseModel
from sillm.args import ModelArgs

########
# References:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py
########
class Model(BaseModel):
    """
    Gemma model wrapper.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        raise NotImplementedError("Gemma model is not implemented yet")