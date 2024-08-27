import unittest

import mlx.core as mx

import sillm

class TestCache(unittest.TestCase):
    def setUp(self):
        self.prompt_cache = sillm.PromptCache(max_size=10)

    def test_lru(self):
        for i in range(1, 11):
            inputs = mx.array([i])
            logits = mx.array([float(i)])
            kv_cache = mx.array([float(i)])

            self.prompt_cache.put(inputs, logits, kv_cache)

        for i in range(1, 5):
            logits, kv_cache = self.prompt_cache.get(mx.array([i]))

            self.assertEqual(logits[0].item(), float(i))
            self.assertEqual(kv_cache[0].item(), float(i))
        
        self.prompt_cache.put(mx.array([99]), mx.array([99.0]), mx.array([99.0]))

        logits, kv_cache = self.prompt_cache.get(mx.array([1]))
        self.assertEqual(logits.item(), 1.0)
        self.assertEqual(kv_cache.item(), 1.0)

        logits, kv_cache = self.prompt_cache.get(mx.array([5]))
        self.assertEqual(logits, None)
        self.assertEqual(kv_cache, None)

if __name__ == '__main__':
    unittest.main()