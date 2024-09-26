import mlx.core as mx
import mlx.nn as nn

from sillm.models.base import BaseModel
from sillm.models.args import ModelArgs
import sillm.models.llama as llama
from sillm.modules.act import init_act

class NemotronLayerNorm1P(nn.LayerNorm):
    """
    Nemotron layer normalization.
    """
    def __call__(self, x):
        weight = self.weight + 1 if "weight" in self else None
        bias = self.bias if "bias" in self else None

        return mx.fast.layer_norm(x, weight, bias, self.eps)

class FeedForward(nn.Module):
    """
    Feed-forward module.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        super().__init__()

        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        
        self.act = init_act(args)
    
    def __call__(self,
                 x: mx.array
                 ) -> mx.array:
        """
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        return self.w2(self.act(self.w3(x)))

class TransformerBlock(llama.TransformerBlock):
    """
    Transformer block.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        nn.Module.__init__(self)
        self.args = args

        self.n_heads = args.n_heads
        self.dim = args.dim

        self.attention = llama.Attention(args=args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = NemotronLayerNorm1P(args.dim, eps=args.norm_eps)
        self.ffn_norm = NemotronLayerNorm1P(args.dim, eps=args.norm_eps)

class Model(llama.Model):
    """
    Nemotron model wrapper.
    """
    def __init__(self, args: ModelArgs):
        """
        Args:
            args: Model arguments.
        """
        BaseModel.__init__(self, args)
        self.args = args

        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = NemotronLayerNorm1P(args.dim, eps=args.norm_eps)

        if args.tie_word_embeddings:
            self.output = None
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)