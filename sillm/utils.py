import os
import resource
import os
import resource

class Memory:
    system_memory = 0

    @classmethod
    def get_process_memory(cls):
        """
        Get process memory usage.
        Returns:
            int: process memory usage in bytes
        """
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    @classmethod
    def get_system_memory(cls):
        """
        Get total system memory.
        Returns:
            int: total system memory in bytes
        """
        if cls.system_memory == 0:
            cls.system_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')

        return cls.system_memory
    
    @classmethod
    def get_memory_usage(cls):
        """
        Get memory usage information for the current process.
        Returns:
            (int, int, float): process memory, system memory, memory utilization
        """
        process_memory = cls.get_process_memory()
        system_memory = cls.get_system_memory()
        memory_utilization = process_memory / float(system_memory)

        return process_memory, system_memory, memory_utilization