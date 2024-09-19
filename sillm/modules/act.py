import logging

import mlx.nn as nn

from sillm.models.args import ModelArgs

logger = logging.getLogger("sillm")

def init_act(args: ModelArgs):
    if args.hidden_act is None:
        logger.debug("No hidden activation specified. Using SiLU.")
        return nn.SiLU()

    if args.hidden_act == "silu":
        return nn.SiLU()
    elif args.hidden_act == "gelu":
        return nn.GELU()
    elif args.hidden_act == "gelu_new":
        return nn.GELU(approx="precise")
    elif args.hidden_act == "gelu_pytorch_tanh":
        return nn.GELU()
    
    logger.warning(f"Unknown hidden activation: {args.hidden_act}. Using SiLU.")
    return nn.SiLU()