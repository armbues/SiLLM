import sys
import os
import logging

import mlx.core as mx

logger = logging.getLogger("sillm")

def init_logger(log_level):
    # Set log level
    logger.setLevel(log_level)

    # Initialize formatter and handler
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def log_arguments(args):
    """
    Log arguments.
    
    Args:
        args: The parsed arguments.
    """
    logger = logging.getLogger("sillm")

    for key, value in args.items():
        logger.debug(f"Argument: {key} = {value}")

def log_memory_usage():
    peak_memory = mx.metal.get_peak_memory()
    system_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    memory_usage = peak_memory / system_memory

    logger.debug(f"Peak memory: {(peak_memory // (1024 ** 2)):,} MB ({memory_usage:.2%} of system memory)")