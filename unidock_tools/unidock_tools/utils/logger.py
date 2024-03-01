import os
import sys
import time
import logging


def init_logging():
    logger = logging.getLogger()
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    formatter = logging.Formatter(
        "[%(levelname)s][%(asctime)s][%(filename)s %(lineno)d] %(message)s"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


def time_logger(func):
    """
    This is a Python decorator function that logs the execution time of a given function.

    Args:
      func: The function that will be decorated with the time_logger decorator.

    Returns:
      The `time_logger` function returns the `wrapper` function, which is a decorator that logs the
    execution time of the decorated function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"[FUNC {func.__name__}]: {end_time - start_time:.2f} seconds")
        return result

    return wrapper
