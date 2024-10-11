import os
import logging

import mlx.core as mx

logger = logging.getLogger("sillm")

def set_memory_limit(memory_limit: float = 0.9,
                     relaxed: bool = False
                     ):
    """
    Set memory limit.
    Args:
        memory_limit: The memory limit as percentage of total system memory.
    """
    # Get system memory
    system_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

    logger.info(f"Setting memory limit to {memory_limit:.0%} of system memory")
    mx.metal.set_memory_limit(int(memory_limit * system_memory), relaxed=relaxed)