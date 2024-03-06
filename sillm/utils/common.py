import logging

import yaml
import psutil

def load_yaml(fpath, args):
    with open(fpath, "r") as f:
        config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(args, key, value)

def log_arguments(args):
    """
    Log arguments.
    
    Args:
        args: The parsed arguments.
    """
    for key, value in args.items():
        logging.debug(f"Argument: {key} = {value}")
    
def log_memory_usage():
    """
    Log memory usage.
    """
    process = psutil.Process()

    process_memory = process.memory_info().rss // (1024 ** 2)
    system_memory = psutil.virtual_memory().total // (1024 ** 2)
    memory_utilization = (process_memory / system_memory) * 100

    logging.debug(f"Memory utilization: {process_memory:,} MB / {system_memory:,} MB ({memory_utilization:.2f}%)")