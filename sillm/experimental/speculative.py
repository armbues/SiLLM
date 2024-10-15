import time
import logging

import mlx.core as mx

import sillm
from sillm.core.llm import LLM
from sillm.models.args import ModelArgs

logger = logging.getLogger("sillm")

class SpeculativeLLM(LLM):
    @staticmethod
    def from_models(draft_llm: LLM, target_llm: LLM):
        return SpeculativeLLM(draft_llm, target_llm)

    def __init__(self, draft_llm: LLM, target_llm: LLM):
        self.draft_model = draft_llm.model
        self.target_model = target_llm.model
        self.tokenizer = draft_llm.tokenizer
        self.args = draft_llm.args

    def generate(self,
                 prompt: str,
                 lookahead: int = 8,
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

        # Initial forward pass through draft and target models
        inputs = mx.array(input_tokens[:-1])[None]
        draft_logits = self.draft_model(inputs, cache=draft_cache)
        target_logits = self.target_model(inputs, cache=target_cache)

        # Calculate evaluation time
        mx.eval(draft_logits, target_logits)
        timing["eval_time"] = time.perf_counter() - start

        # Main generation loop
        tokens = input_tokens
        text = self.tokenizer.decode(tokens)
        while True:
            # Draft speculative tokens
            y = mx.array(tokens[-1:])
            draft_tokens = []
            for _ in range(lookahead):
                draft_logits = self.draft_model(y[None], cache=draft_cache)
                draft_logits = draft_logits[:, -1, :]

                y = sample(draft_logits)
                draft_tokens.append(y.item())

                if y.item() in stop_tokens:
                    break
            draft_indices = mx.array(draft_tokens)
            
            # Check drafted tokens with the target model
            inputs = mx.array(tokens[-1:] + draft_tokens[:-1])[None]
            target_logits = self.target_model(inputs, cache=target_cache)

            # Accept or reject draft tokens
            target_indices = sample(target_logits).flatten()
            target_tokens = target_indices.tolist()
            accept_mask = (target_indices == draft_indices).tolist()
            num_accepted = (accept_mask + [False]).index(False)

            if num_accepted == 0:
                # All draft tokens were rejected - sample one token from target model
                result_tokens = [target_tokens[0]]
                speculative["drafted"] += [False]
            elif num_accepted < len(draft_tokens):
                # Some draft tokens were accepted - sample last token from target model
                result_tokens = draft_tokens[:num_accepted] + [target_tokens[num_accepted]]
                speculative["drafted"] += [True] * num_accepted + [False]
            else:
                # All draft tokens were accepted
                result_tokens = draft_tokens
                speculative["drafted"] += [True] * num_accepted

            # Check for completion conditions
            for i, t in enumerate(result_tokens):
                if len(tokens) + i >= max_tokens:
                    metadata["finish_reason"] = "length"
                    result_tokens = result_tokens[:i]
                if t in stop_tokens:
                    metadata["finish_reason"] = "stop"
                    result_tokens = result_tokens[:i]

            # Update tokens
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
                dc.offset = len(tokens) - 1
                tc.offset = len(tokens) - 1

            # Update text
            text_offset = len(text)
            text = self.tokenizer.decode(tokens)

            # Yield result
            yield text[text_offset:], metadata

            # Check for completion
            if "finish_reason" in metadata:
                break

class SpeculativeEdit(LLM):
    @staticmethod
    def from_model(llm: LLM):
        return SpeculativeEdit(llm)

    def __init__(self, model: LLM):
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.args = model.args

        self._quantization = model._quantization
        self._mapping = model._mapping

    def generate(self,
                 prompt: str,
                 target: str,
                 lookahead: int = 32,
                 key_size: int = 3,
                 temperature: float = 0.0,
                 max_tokens: int = 8192,
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
        cache = sillm.KVCache.for_model(self.model)

        # Initial forward pass
        inputs = mx.array(input_tokens[:-1])[None]
        logits = self.model(inputs, cache=cache)

        # Calculate evaluation time
        mx.eval(logits)
        timing["eval_time"] = time.perf_counter() - start

        # Initialize speculative mode
        target_tokens = input_tokens[-1:] + self.tokenizer.encode(target, bos=False, eos=True) + [self.tokenizer.eos_id] * lookahead
        spec_offset = 0
        spec_mode = True
        total_accepted = 0

        # Main generation loop
        tokens = input_tokens
        text = self.tokenizer.decode(tokens)
        while True:
            if spec_mode and lookahead > 1:
                # Speculate based on target text
                inputs = mx.array(target_tokens[spec_offset:spec_offset+lookahead])[None]
                logits = self.model(inputs, cache=cache)

                spec_tokens = mx.array(target_tokens[spec_offset+1:spec_offset+lookahead+1])
                edit_tokens = mx.argmax(logits, axis=-1).flatten()[:len(spec_tokens)]
                accept_mask = (spec_tokens == edit_tokens).tolist()
                num_accepted = (accept_mask + [False]).index(False)

                if num_accepted == 0:
                    # All target tokens were rejected - sample first token from the model
                    logits = logits[:, 0, :]
                    y = sample(logits)

                    result_tokens = [y.item()]
                    speculative["drafted"] += [False]
                elif num_accepted < lookahead:
                    # Some target tokens were accepted - sample last token from the model
                    result_tokens = spec_tokens[:num_accepted].tolist() + [edit_tokens[num_accepted].item()]
                    speculative["drafted"] += [True] * num_accepted + [False]
                else:
                    # All target tokens were accepted
                    result_tokens = spec_tokens[:num_accepted].tolist()
                    speculative["drafted"] += [True] * num_accepted

                # Update speculative mode
                spec_mode = (num_accepted == lookahead)
                spec_offset += num_accepted
            else:
                # Generate next token
                inputs = mx.array(tokens[-1:])[None]
                logits = self.model(inputs, cache=cache)
                y = sample(logits)

                result_tokens = [y.item()]
                speculative["drafted"] += [False]
                num_accepted = 0

                # Try to recover speculative mode
                spec_key = tokens[-key_size:]
                for i in range(total_accepted, len(target_tokens) - lookahead - key_size):
                    if target_tokens[i:i+key_size] == spec_key:
                        spec_offset = i + key_size
                        spec_mode = True

            # Check for completion conditions
            for i, t in enumerate(result_tokens):
                if len(tokens) + i >= max_tokens:
                    metadata["finish_reason"] = "length"
                    result_tokens = result_tokens[:i]
                if t in stop_tokens:
                    metadata["finish_reason"] = "stop"
                    result_tokens = result_tokens[:i]

            # Update tokens
            tokens += result_tokens
            total_accepted += num_accepted
            
            # Update metadata
            usage["completion_tokens"] += len(result_tokens)
            usage["total_tokens"] = len(tokens)
            timing["runtime"] = time.perf_counter() - start
            speculative["num_accepted"] = total_accepted
            if token_ids:
                metadata["token_ids"] = tokens

            # Adjust cache offsets
            for c in cache:
                c.offset = len(tokens) - 1

            # Update text
            text_offset = len(text)
            text = self.tokenizer.decode(tokens)

            # Yield result
            yield text[text_offset:], metadata

            # Check for completion
            if "finish_reason" in metadata:
                break