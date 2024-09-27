from functools import partial

import mlx.core as mx
import mlx.nn as nn

@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_k(logits: mx.array,
          k: int = 1
          ) -> mx.array:
    top_k_indices = mx.argpartition(logits, k, axis=-1)[:, -k:]
    mask = mx.zeros_like(logits)
    mask[:, top_k_indices] = 1.0

    return logits * mask

@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_p(logits: mx.array,
          p: float = 0.9
          ) -> mx.array:
    probs = mx.softmax(logits, axis=-1)
    sorted_indices = mx.argsort(probs, axis=-1).squeeze(0)
    sorted_probs = probs[..., sorted_indices]
    cum_probs = mx.cumsum(sorted_probs, axis=-1)
    
    mask = cum_probs <= p
    logits = mx.where(mask[..., sorted_indices], logits, 0.0)

    return logits

@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
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