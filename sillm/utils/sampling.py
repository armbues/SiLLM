import string

import mlx.core as mx

def ascii_token_logit_mask(
        tokenizer,
        output_size: int
        ) -> mx.array:
    """
    Create a logit mask filtering out tokens with non-ASCII printable characters.
    """
    mask = mx.zeros(output_size)

    for token, i in tokenizer.vocab.items():
        if all(c in string.printable for c in token.strip('Ġ')):
            mask[i] = 1.0

    return mask