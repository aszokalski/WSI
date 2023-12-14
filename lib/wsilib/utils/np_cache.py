from functools import wraps, lru_cache
import numpy as np


def hashable_array(arr):
    if isinstance(arr, np.ndarray):
        # check dimentions
        if len(arr.shape) == 1:
            return tuple(arr)
        else:
            return tuple(map(tuple, arr))
    return arr


def np_cache(function):
    @lru_cache()
    def cached_wrapper(*hashable_args, **hashable_kwargs):
        args = [
            np.array(arg) if isinstance(arg, tuple) else arg for arg in hashable_args
        ]
        kwargs = {
            k: np.array(v) if isinstance(v, tuple) else v
            for k, v in hashable_kwargs.items()
        }
        return function(*args, **kwargs)

    @wraps(function)
    def wrapper(*args, **kwargs):
        hashable_args = [hashable_array(arg) for arg in args]
        hashable_kwargs = {k: hashable_array(v) for k, v in kwargs.items()}
        return cached_wrapper(*hashable_args, **hashable_kwargs)

    # Copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
