import os
import resource
import logging

def get_process_memory() -> int:
    """
    Get process memory usage.
    Returns:
        int: process memory usage in bytes
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def get_system_memory() -> int:
    """
    Get total system memory.
    Returns:
        int: total system memory in bytes
    """
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    
def log_memory_usage():
    """
    Log memory usage.
    """
    process_memory = get_process_memory() // (1024 * 1024)
    system_memory = get_system_memory() // (1024 * 1024)
    memory_utilization = 100 * process_memory / float(system_memory)

    logging.debug(f"Memory utilization: {process_memory:,} MB / {system_memory:,} MB ({memory_utilization:.2f}%)")