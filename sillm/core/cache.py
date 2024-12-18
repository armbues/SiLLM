import collections
import hashlib
import logging

from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map

logger = logging.getLogger("sillm")

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/fc93c557238e9441835afe2748fd170016cb068b/llms/mlx_lm/models/base.py#L11
########
class KVCache:
    @staticmethod
    def for_model(model,
                  step : int = 256
                  ):
        return [KVCache(step) for _ in range(model.args.n_layers)]

    def __init__(self,
                 step : int = 256
                 ):
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = step

    def update_and_fetch(self,
                         keys,
                         values
                         ):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev_offset = self.offset
        
        if self.keys is None or (prev_offset + num_steps) > self.keys.shape[2]:
            new_steps = (self.step + num_steps - 1) // self.step

            k_shape = (B, n_kv_heads, new_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, new_steps * self.step, v_head_dim)

            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            
            if self.keys is not None:
                if prev_offset % self.step != 0:
                    self.keys = self.keys[..., :prev_offset, :]
                    self.values = self.values[..., :prev_offset, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += num_steps

        self.keys[..., prev_offset : self.offset, :] = keys
        self.values[..., prev_offset : self.offset, :] = values

        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
    
    def copy(self):
        new = KVCache(self.step)
        
        if self.keys is not None:
            new.keys = mx.array(self.keys)
        if self.values is not None:
            new.values = mx.array(self.values)
        new.offset = self.offset

        return new

    @property
    def state(self):
        return self.keys, self.values
    
    def quantize(self,
                 group_size: int = 32,
                 bits: int = 4
                 ):
        """
        Quantize the KV cache.
        
        """
        cache = QuantizedKVCache(self.step, group_size, bits)
        cache.offset = self.offset
        if self.keys is not None:
            cache.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
        if self.values is not None:
            cache.values = mx.quantize(self.values, group_size=group_size, bits=bits)

        return cache

class QuantizedKVCache(KVCache):
    @staticmethod
    def for_model(model,
                  step : int = 256,
                  group_size: int = 32,
                  bits: int = 4,
                  ):
        return [QuantizedKVCache(step, group_size, bits) for _ in range(model.args.n_layers)]
    
    def __init__(self,
                 step : int = 256,
                 group_size: int = 32,
                 bits: int = 4,
                 ):
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = step

        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self,
                         keys,
                         values
                         ):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev_offset = self.offset
        
        if self.keys is None or (prev_offset + num_steps) > self.keys[0].shape[2]:
            new_steps = (self.step + num_steps - 1) // self.step

            if self.keys is None:
                k_shape = (B, n_kv_heads, new_steps * self.step, k_head_dim)
                v_shape = (B, n_kv_heads, new_steps * self.step, v_head_dim)

                self.keys = mx.quantize(mx.zeros(k_shape, keys.dtype), group_size=self.group_size, bits=self.bits)
                self.values = mx.quantize(mx.zeros(v_shape, keys.dtype), group_size=self.group_size, bits=self.bits)
            else:
                if prev_offset % self.step != 0:
                    self.keys, self.values = tree_map(lambda x: x[..., :prev_offset, :], (self.keys, self.values))

                def expand(x, shape):
                    new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                    return mx.concatenate([x, new_x], axis=-2)

                shape = (B, n_kv_heads, new_steps * self.step)
                self.keys, self.values = tree_map(lambda x: expand(x, shape), (self.keys, self.values))

        self.offset += num_steps

        keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        values = mx.quantize(values, group_size=self.group_size, bits=self.bits)

        for i in range(len(self.keys)):
            self.keys[i][..., prev_offset : self.offset, :] = keys[i]
            self.values[i][..., prev_offset : self.offset, :] = values[i]

        return tree_map(lambda x: x[..., : self.offset, :], (self.keys, self.values))

class PromptCache():
    """
    Cache for prompt logits and KV cache.
    """
    def __init__(self,
                 max_size: int = 10
                 ):
        self.logits = {}
        self.kv_cache = {}

        self.lru = collections.OrderedDict()

        self.max_size = max_size

        # TODO make multi-thread safe?

    def _key(self,
             inputs: mx.array
             ):
        return hashlib.sha256(inputs).hexdigest()
    
    def put(self,
            inputs: mx.array,
            logits: mx.array,
            kv_cache: Optional[KVCache] = None,
            ):
        """
        Add cache entry.
        """
        key = self._key(inputs)

        logger.debug(f"Adding prompt cache entry for key {key}")

        if key not in self.lru:
            self.logits[key] = logits
            self.kv_cache[key] = [c.copy() for c in kv_cache]
            self.lru[key] = 0

        if len(self.lru) > self.max_size:
            pop_key, _ = self.lru.popitem(last=False)
            self.logits.pop(pop_key)
            self.kv_cache.pop(pop_key)

    def get(self,
            inputs: mx.array|str
            ):
        """
        Get cache entry.
        """
        if type(inputs) is mx.array:
            key = self._key(inputs)
        else:
            key = inputs

        if key in self.lru:
            # Update LRU
            self.lru.move_to_end(key)
            self.lru[key] += 1

            logits = self.logits[key]
            kv_cache = [c.copy() for c in self.kv_cache[key]]

            logger.debug(f"Retrieving prompt cache entry for key {key}")
            
            return logits, kv_cache
        
        return None, None