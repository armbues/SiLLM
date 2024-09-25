import string

import mlx.core as mx
import mlx.nn as nn

def sample(logits: mx.array,
           temperature: float = 0.0,
           logprobs: bool = False
           ):
    if temperature == 0:
        y = mx.argmax(logits, axis=-1)
    else:
        y = mx.random.categorical(logits * (1 / temperature))

    p = 0.0
    if logprobs:
        p = nn.log_softmax(logits, axis=-1)[0,y].item()
    # TODO add top-p sampling

    return y, p

def apply_repetition_penalty(logits: mx.array,
                             tokens: list,
                             repetition_penalty: float = 1.0,
                             repetition_window: int = 25
                             ) -> mx.array:
    indices = mx.array(tokens[-repetition_window:])
    repeated_logits = logits[:, indices]
    repeated_logits = mx.where(repeated_logits < 0, repeated_logits * repetition_penalty, repeated_logits / repetition_penalty)
    logits[:, indices] = repeated_logits

    return logits

def ascii_token_logit_mask(
        tokenizer,
        output_size: int
        ) -> mx.array:
    """
    Create a logit mask filtering out tokens with non-ASCII printable characters.
    """
    mask = mx.zeros(output_size)

    for i, s in enumerate(tokenizer.vocab_strings):
        if all(c in string.printable for c in s.strip()):
            mask[i] = 1.0

    for i in tokenizer.special_ids:
        mask[i] = 1.0

    return mask