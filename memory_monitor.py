import psutil
import logging

logger = logging.getLogger(__name__)

def log_memory_usage(stage: str):
    """Log current memory usage"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"{stage}: Memory usage: {memory_mb:.1f} MB")
    return memory_mb

def check_memory_limit(limit_mb: int = 450):
    """Check if approaching memory limit"""
    memory_mb = log_memory_usage("Memory Check")
    if memory_mb > limit_mb:
        logger.warning(f"Memory usage ({memory_mb:.1f} MB) approaching limit ({limit_mb} MB)")
    return memory_mb < limit_mb