import string
import re

import mlx.core as mx

class LogitFilter:
    """
    Logit mask filter.
    """
    def __init__(self,
                 tokenizer,
                 output_size: int
                 ):
        self.tokenizer = tokenizer
        self.output_size = output_size

    def reset(self):
        pass

    def __call__(self,
                 logits: mx.array
                 ) -> mx.array:
        raise NotImplementedError("Class LogitFilter is used for inheritance only")

class ASCIIFilter(LogitFilter):
    """
    Static logit mask filtering out tokens with non-ASCII printable characters.
    """
    def __init__(self,
                 tokenizer,
                 output_size: int
                 ):
        mask = mx.zeros(output_size)
        for i, s in enumerate(tokenizer.vocab_strings):
            if all(c in string.printable for c in s.strip()):
                mask[i] = 1.0

        for i in tokenizer.special_ids:
            mask[i] = 1.0

        self.mask = mask

    def __call__(self,
                 logits: mx.array
                 ) -> mx.array:
        return logits * self.mask