import logging
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

logger = logging.getLogger("sillm")

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder
    References:
    https://arxiv.org/pdf/2309.08600
    https://arxiv.org/pdf/2406.04093v1
    """
    def __init__(self,
                 dim: int,
                 hidden_dim: int
                 ):
        super().__init__()

        self.num_features = hidden_dim
        logger.info(f"Initialized Sparse Autoencoder with input dimensions {dim} and {hidden_dim} features")

    def encode(self, x: mx.array):
        raise NotImplementedError("Method encode is not implemented")
    
    def decode(self, x: mx.array):
        raise NotImplementedError("Method decode is not implemented")
    
    def save_weights(self,
                     weights_path: str
                     ):
        """
        Save model weights into a single safetensors file.
        Args:
            weights_path: Path to weights file.
        """
        state = dict(tree_flatten(self.model.parameters()))
        mx.save_safetensors(weights_path, state)

        logger.info(f"Saved weights for Sparse Autoencoder to {weights_path}")

    @staticmethod
    def load(model_path: str):
        """
        Initialize sparse autoencoder from weights file
        Args:
            args: LLM config arguments
            model_path: Path to sparse autoencoder weights
        """
        weights = mx.load(model_path)

        if "threshold" in weights:
            hidden_dim, dim = weights["W_enc"].shape
            
            model = JumpReLUSparseAutoencoder(dim, hidden_dim)
        elif "encoder.weight" in weights:
            hidden_dim, dim = weights["encoder.weight"].shape

            model = EleutherSparseAutoencoder(dim, hidden_dim)
        else:
            raise ValueError("Unknown type of Sparse Autoencoder")
        
        model.update(weights)
        logger.info(f"Loaded weights for Sparse Autoencoder from {model_path}")

        return model

class JumpReLUSparseAutoencoder(SparseAutoencoder):
    """
    JumpReLU Sparse Autoencoder
    References:
    https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf
    https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp
    """
    def __init__(self,
                 dim: int,
                 hidden_dim: int
                 ):
        super().__init__(dim, hidden_dim)

        scale = math.sqrt(1.0 / hidden_dim)
        self.W_dec = mx.random.uniform(-scale, scale, (hidden_dim, dim))
        self.b_dec = mx.zeros((dim,))

        self.W_enc = self.W_dec.T
        self.b_enc = mx.zeros((hidden_dim,))

        self.threshold = mx.full((hidden_dim), 0.001)

    def encode(self, x: mx.array):
        x = x @ self.W_enc + self.b_enc
        mask = (x > self.threshold)

        return mask * nn.relu(x)
    
    def decode(self, x: mx.array):
        return x @ self.W_dec + self.b_dec

class EleutherSparseAutoencoder(SparseAutoencoder):
    """
    Eleuther Sparse Autoencoder
    References:
    https://github.com/EleutherAI/sae
    """
    def __init__(self,
                 dim: int,
                 hidden_dim: int
                 ):
        super().__init__(dim, hidden_dim)

        self.encoder = nn.Linear(dim, hidden_dim)

        scale = math.sqrt(1.0 / hidden_dim)
        self.W_dec = mx.random.uniform(-scale, scale, (hidden_dim, dim))
        self.b_dec = mx.zeros((dim,))

    def encode(self, x: mx.array):
        x = x - self.b_dec
        x = self.encoder(x)
        
        return nn.relu(x)