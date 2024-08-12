import logging
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from sillm.models.args import ModelArgs

logger = logging.getLogger("sillm")

class SparseAutoencoder(nn.Module):
    """
    ReLU Sparse Autoencoder
    References:
    https://arxiv.org/pdf/2309.08600
    https://github.com/openai/sparse_autoencoder/
    https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp
    """
    def __init__(self,
                 dim: int,
                 hidden_dim: int
                 ):
        super().__init__()

        scale = math.sqrt(1.0 / dim)
        self.W_enc = mx.random.uniform(-scale, scale, (dim, hidden_dim))
        self.b_enc = mx.random.uniform(-scale, scale, (hidden_dim,))

        self.threshold = mx.zeros((hidden_dim))

        scale = math.sqrt(1.0 / hidden_dim)
        self.W_dec = mx.random.uniform(-scale, scale, (hidden_dim, dim))
        self.b_dec = mx.random.uniform(-scale, scale, (dim,))

    def encode(self, x: mx.array):
        x = x @ self.W_enc + self.b_enc
        mask = (x > self.threshold)

        return mask * nn.relu(x)
    
    def decode(self, x: mx.array):
        return x @ self.W_dec + self.b_dec
    
    def forward(self, x: mx.array):
        x = self.encode(x)

        return self.decode(x)
    
    def loss(self, inputs: mx.array):
        latents = self.encode(inputs)
        reconstruction = self.decode(latents)

        mse = nn.losses.mse_loss(reconstruction, inputs)
        sparsity = nn.losses.l1_loss(latents, inputs)

        return mse + sparsity
    
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
    def load(args: ModelArgs,
             model_path: str
             ):
        """
        Initialize sparse autoencoder from weights file
        Args:
            args: LLM config arguments
            model_path: Path to sparse autoencoder weights
        """
        weights = mx.load(model_path)
        hidden_dim = weights["W_enc"].shape[1]
        
        model = SparseAutoencoder(args.dim, hidden_dim)
        model.update(weights)

        logger.info(f"Loaded weights for Sparse Autoencoder with {args.dim}/{hidden_dim} dimensions")

        return model