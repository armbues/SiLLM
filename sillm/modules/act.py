import logging

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from sillm.models.args import ModelArgs

logger = logging.getLogger("sillm")

@partial(mx.compile, shapeless=True)
def relu2(x):
    return nn.relu(x).square()

def init_act(args: ModelArgs):
    if args.hidden_act is None:
        logger.debug("No hidden activation specified. Using SiLU.")
        return nn.SiLU()

    if args.hidden_act == "silu":
        return nn.silu
    elif args.hidden_act == "gelu":
        return nn.gelu
    elif args.hidden_act == "gelu_new":
        return nn.gelu
    elif args.hidden_act == "gelu_pytorch_tanh":
        return nn.gelu
    elif args.hidden_act == "relu2":
        return relu2
    
    logger.warning(f"Unknown hidden activation: {args.hidden_act}. Using SiLU.")
    return nn.SiLU()