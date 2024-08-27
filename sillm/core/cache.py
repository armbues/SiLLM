import collections
import hashlib

import mlx.core as mx

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
            kv_cache: mx.array
            ):
        """
        Add cache entry.
        """
        key = self._key(inputs)

        if key not in self.lru:
            self.logits[key] = logits
            self.kv_cache[key] = kv_cache
            self.lru[key] = 0

        if len(self.lru) > self.max_size:
            pop_key, _ = self.lru.popitem(last=False)
            self.logits.pop(pop_key)
            self.kv_cache.pop(pop_key)

    def get(self,
            inputs: mx.array
            ):
        """
        Get cache entry.
        """
        key = self._key(inputs)

        if key in self.lru:
            self.lru.move_to_end(key)
            self.lru[key] += 1
            
            return self.logits[key], self.kv_cache[key]
        
        return None, None