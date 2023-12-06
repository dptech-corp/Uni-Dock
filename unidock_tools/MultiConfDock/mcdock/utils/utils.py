################## makedirs ##################
import datetime, random, string, os
from pathlib import Path
from functools import wraps
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s][%(levelname)s]%(message)s',
)


def generate_random_string(length:int=4) -> str:
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def makedirs(
    prefix: str = "results",
    date: bool = True,
    randomID: bool = False
) -> Path:
    name = prefix
    if date:
        now = datetime.datetime.now()
        date = now.strftime('%Y%m%d%H%M%S')
        name += f"_{date}"
    if randomID:
        name += f"_{generate_random_string(8)}"
    os.makedirs(name, exist_ok=True)
    return Path(name)
################## makedirs ##################

################## time logger ##################
import time
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
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
################## time logger ##################