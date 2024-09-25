import time
import logging

import numpy as np

import mlx.core as mx

import sillm
from sillm.core.llm import LLM

logger = logging.getLogger("sillm")

class SpeculativeLLM(LLM):
    @staticmethod
    def from_models(draft_llm: LLM, target_llm: LLM):
        return SpeculativeLLM(draft_llm.model, target_llm.model, draft_llm.tokenizer)

    def __init__(self,
                 draft_model,
                 target_model,
                 tokenizer
                 ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer

    def generate(self,
                 prompt: str,
                 lookahead: int = 4,
                 temperature: float = 0.0,
                 max_tokens: int = 2048,
                 token_ids: bool = False,
                 ):
        start = time.perf_counter()
        
        # Tokenize prompt
        input_tokens = self.tokenizer.encode(prompt)
            
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
        speculative = {
            "num_accepted": 0,
            "drafted": []
        }
        metadata = {
            "timing": timing,
            "usage": usage,
            "speculative": speculative,
            "token_ids": []
        }

        # Define stop tokens
        stop_tokens = self.tokenizer.special_ids

        def sample(logits):
            if temperature == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temperature))

        # Initialize KV caches
        draft_cache = sillm.KVCache.for_model(self.draft_model)
        target_cache = sillm.KVCache.for_model(self.target_model)

        # Main generation loop
        tokens, text = [], ""
        while True:
            if len(tokens) == 0:
                # Initial forward pass through draft and target models
                inputs = mx.array(input_tokens[:-1])[None]
                draft_logits = self.draft_model(inputs, cache=draft_cache)
                target_logits = self.target_model(inputs, cache=target_cache)

                mx.async_eval(draft_logits, target_logits)
                timing["eval_time"] = time.perf_counter() - start
                
                tokens = input_tokens
                text = self.tokenizer.decode(tokens)
            
            # Draft tokens
            y = mx.array(tokens[-1:])
            draft_tokens = []
            for _ in range(lookahead):
                # Forward pass through draft model
                logits = self.draft_model(y[None], cache=draft_cache)
                logits = logits[:, -1, :]

                # Sample draft token
                y = sample(logits)
                draft_tokens.append(y.item())
            draft_indices = mx.array(draft_tokens)
            
            # Forward pass drafted tokens through the target model
            inputs = mx.array(tokens[-1:] + draft_tokens[:-1])[None]
            target_logits = self.target_model(inputs, cache=target_cache)

            # Accept or reject drafted tokens
            logits = target_logits[:, -lookahead:, :]
            target_indices = sample(logits).flatten()
            accept_mask = (target_indices == draft_indices)
            num_accepted = (accept_mask.tolist() + [False]).index(False)

            if num_accepted == 0:
                # All draft tokens were rejected - sample one token from target model
                logits = target_logits[:, -lookahead, :]
                y = sample(logits)

                result_tokens = [y.item()]
                speculative["drafted"] += [False]
            elif num_accepted < lookahead:
                # Some draft tokens were accepted - sample last token from target model
                result_tokens = draft_tokens[:num_accepted] + [target_indices[num_accepted].item()]
                speculative["drafted"] += [True] * num_accepted + [False]
            else:
                # All draft tokens were accepted
                result_tokens = draft_tokens[:num_accepted]
                speculative["drafted"] += [True] * num_accepted

            # Check for completion conditions
            for i, t in enumerate(result_tokens):
                if len(tokens) + i >= max_tokens:
                    metadata["finish_reason"] = "length"
                    result_tokens = result_tokens[:i]
                if t in stop_tokens:
                    metadata["finish_reason"] = "stop"
                    result_tokens = result_tokens[:i]
            tokens += result_tokens

            # Update metadata
            usage["completion_tokens"] += len(result_tokens)
            usage["total_tokens"] = len(tokens)
            timing["runtime"] = time.perf_counter() - start
            speculative["num_accepted"] += num_accepted
            if token_ids:
                metadata["token_ids"] = tokens

            # Adjust cache offsets
            for dc, tc in zip(draft_cache, target_cache):
                dc.offset = len(tokens)
                tc.offset = len(tokens)

            text_offset = len(text)
            text = self.tokenizer.decode(tokens)

            # Yield result
            yield text[text_offset:], metadata

            # Check for completion
            if "finish_reason" in metadata:
                break