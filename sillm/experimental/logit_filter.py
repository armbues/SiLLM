import logging
import string
import re

import mlx.core as mx

logger = logging.getLogger("sillm")

class LogitFilter:
    """
    Logit mask filter.
    """
    def __init__(self):
        raise NotImplementedError("Class LogitFilter is used for inheritance only")
    
    def update(self,
               metadata: dict,
               tokens: list
               ):
        pass

    def __call__(self,
                 logits: mx.array
                 ) -> mx.array:
        return logits * self.mask

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
    
class RegexFilter(LogitFilter):
    """
    Static logit mask filtering out tokens matching a regular expression.
    """
    def __init__(self,
                 tokenizer,
                 output_size: int,
                 regex: str
                 ):
        mask = mx.ones(output_size)
        for i, s in enumerate(tokenizer.vocab_strings):
            if re.match(regex, s):
                mask[i] = 0.0

        for i in tokenizer.special_ids:
            mask[i] = 1.0

        self.mask = mask
    
class MinCompletionFilter(LogitFilter):
    """
    Logit filter preventing stop tokens before a minimum completion length.
    """
    def __init__(self,
                 tokenizer,
                 output_size: int,
                 min_length: int,
                 stop_tokens: list = None
                 ):
        self.output_size = output_size
        self.min_length = min_length

        if stop_tokens is None:
            stop_tokens = tokenizer.special_ids
        self.stop_tokens = [tokenizer.encode(token, bos=False)[0] if isinstance(token, str) else token for token in stop_tokens]

        mask = mx.ones(output_size)
        for token in self.stop_tokens:
            mask[token] = 0.0

        self.mask = mask

    def update(self,
               metadata: dict,
               tokens: list
               ):
        if metadata["usage"]["completion_tokens"] >= self.min_length:
            self.mask = mx.ones(self.output_size)

class MinReasoningFilter(MinCompletionFilter):
    """
    Logit filter preventing end of reasoning tokens before a minimum completion length.
    """
    def __init__(self,
                 tokenizer,
                 output_size: int,
                 min_length: int,
                 stop_tokens: list = ["</think>"]
                 ):
        super().__init__(tokenizer, output_size, min_length, stop_tokens)