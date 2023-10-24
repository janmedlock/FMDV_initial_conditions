'''Utilities for measuring memory usage of code.'''

import functools
import tracemalloc

import _common


def tracer(func):
    '''A decorator to measure memory usage of function calls.'''
    @functools.wraps(func)
    def wrapped(*args, **kwds):
        tracemalloc.start()
        result = func(*args, **kwds)
        (size, peak) = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        call_str = _common.call_to_str(func, *args, **kwds)
        peak_str = f'{peak/1024**3:.1f}GB'
        print(f'{call_str}: {peak_str} peak memory used')
        return result
    return wrapped
