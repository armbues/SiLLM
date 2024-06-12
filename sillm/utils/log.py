import sys
import os
import logging

import mlx.core as mx

logger = logging.getLogger("sillm")

def init_logger(log_level,
                fmt = "%(asctime)s %(levelname)s %(message)s",
                datefmt = "%Y-%m-%d %H:%M:%S",
                add_stdout = True
                ):
    """
    Initialize the logger.
    Args:
        log_level: The log level.
        fmt: The log format.
        datefmt: The date format.
    Returns:
        The logger.
    """
    # Set log level
    logger.setLevel(log_level)

    if add_stdout:
        # Initialize formatter and handler
        formatter = logging.Formatter(fmt, datefmt=datefmt)
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

def log_memory_usage(reset=False):
    peak_memory = mx.metal.get_peak_memory()
    system_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    memory_usage = peak_memory / system_memory

    logger.debug(f"Peak memory: {(peak_memory // (1024 ** 2)):,} MB ({memory_usage:.2%} of system memory)")

    if reset:
        mx.metal.reset_peak_memory()