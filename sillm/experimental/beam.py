import time

import mlx.core as mx
import mlx.nn as nn

from sillm.core.llm import LLM
from sillm.core.cache import KVCache

def beam_search(model: LLM,
                prompt: str,
                beam_width: int = 4,
                min_choices: int = 1,
                length_penalty: float = 0.6,
                max_tokens: int = 8192,
                ):
    tokenizer = model.tokenizer
    
    start = time.perf_counter()

    # Tokenize prompt
    input_tokens = tokenizer.encode(prompt)
    
    # Initialize metadata
    timing = {
        "runtime": 0.0,
        "eval_time": 0.0,
        "tokenizer_time": time.perf_counter() - start
    }
    usage = {
        "prompt_tokens": len(input_tokens),
        "completion_tokens": 0,
        "total_tokens": 0
    }
    metadata = {
        "timing": timing,
        "usage": usage,
        "logprobs": [],
        "token_ids": [],
        "finish_reason": "length"
    }

    # Define stop tokens
    stop_tokens = tokenizer.special_ids

    # Adjust max tokens
    max_tokens = max_tokens - len(input_tokens)

    # Initialize KV caches
    cache = KVCache.for_model(model)

    # Initial forward pass
    inputs = mx.array(input_tokens)[None]
    logits = model.model(inputs, cache=cache)

    timing["eval_time"] = time.perf_counter() - start

    # First round of beam updates
    logits = logits[:, -1, :]
    logits = nn.log_softmax(logits, axis=-1)
    top_k_indices = mx.argpartition(logits, beam_width, axis=-1)[:, -beam_width:]
    beam_logprobs = mx.take_along_axis(logits, top_k_indices, axis=-1).squeeze(0).reshape(beam_width, 1, 1)

    beam_tokens = [[] for _ in range(beam_width)]
    for i in range(beam_width):
        token_id = top_k_indices[0, i].item()
        beam_tokens[i].append(token_id)

    # Initialize cache for beam width by repeating keys and values
    for layer_cache in cache:
        layer_cache.keys = mx.repeat(layer_cache.keys, repeats=beam_width, axis=0)
        layer_cache.values = mx.repeat(layer_cache.values, repeats=beam_width, axis=0)

    finished_beams = []
    finished_scores = []
    while True:
        inputs = mx.array(beam_tokens)[:, -1:]
        logits = model.model(inputs, cache=cache)
        logits = nn.log_softmax(logits, axis=-1)

        top_k_indices = mx.argpartition(logits, beam_width, axis=-1)[:, :, -beam_width:]
        top_k_logprobs = mx.take_along_axis(logits, top_k_indices, axis=-1)
        
        combined_logprobs = top_k_logprobs + beam_logprobs
        flat_indices = top_k_indices.reshape(-1)
        flat_scores = combined_logprobs.reshape(-1)

        result_indices = mx.argpartition(flat_scores, beam_width, axis=-1)
        beam_indices = result_indices // beam_width
        considered_tokens = mx.take(flat_indices, result_indices).tolist()

        new_beam_tokens = []
        cache_indices = []
        for i in range(len(result_indices)-1, 0, -1):
            beam_index = beam_indices[i].item()
            token = considered_tokens[i]
            tokens = beam_tokens[beam_index] + [token]
            
            logprob = flat_scores[result_indices[i]]
            # Normalize logprob with length penalty
            if length_penalty > 0.0:
                logprob /= len(tokens) ** length_penalty
            
            if token in stop_tokens:
                # Add completion to finished beams
                finished_beams.append(tokens)
                finished_scores.append(logprob.item())
            else:
                # Update active beams
                beam_logprobs[len(new_beam_tokens)] = logprob
                new_beam_tokens.append(tokens)
                cache_indices.append(beam_index)

            # Found enough new beams
            if len(new_beam_tokens) >= beam_width:
                break
        beam_tokens = new_beam_tokens

        # Stop if not enough active beams
        if len(new_beam_tokens) < beam_width:
            break

        # Stop if max tokens reached
        if len(beam_tokens[0]) >= max_tokens:
            finished_beams.extend(beam_tokens)
            finished_scores.extend(beam_logprobs.flatten().tolist())
            break

        # Stop if best beam is worse than best finished beam
        if len(finished_scores) >= min_choices and max(beam_logprobs) < max(finished_scores):
            break

        # Update cache
        new_cache = KVCache.for_model(model.model)
        cache_indices = mx.array(cache_indices)
        for l in range(len(cache)):
            new_cache[l].offset = cache[l].offset
            new_cache[l].keys = mx.take(cache[l].keys, cache_indices, axis=0)
            new_cache[l].values = mx.take(cache[l].values, cache_indices, axis=0)
        cache = new_cache

    metadata["choices"] = []
    for i in mx.argsort(mx.array(finished_scores)).tolist()[::-1]:
        finish_reason = "length"
        if finished_beams[i][-1] in stop_tokens:
            finished_beams[i] = finished_beams[i][:-1]
            finish_reason = "stop"
        else:
            finish_reason = "length"

        choice = {
            "text": tokenizer.decode(finished_beams[i]),
            "score": finished_scores[i],
            "finish_reason": finish_reason
        }
        metadata["choices"].append(choice)

    result_index = mx.argmax(mx.array(finished_scores)).item()
    result_tokens = finished_beams[result_index]
    result = tokenizer.decode(result_tokens)

    timing["runtime"] = time.perf_counter() - start
    usage["completion_tokens"] = len(result_tokens)
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    return result, metadata