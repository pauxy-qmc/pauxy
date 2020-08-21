import time
from functools import wraps

def timer(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        return result
    return wrapper
