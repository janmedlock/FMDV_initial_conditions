'''Utilities for timing code.'''

import functools
import time

import _common


def timer(func):
    '''A decorator to time function calls.'''
    @functools.wraps(func)
    def wrapped(*args, **kwds):
        start = time.perf_counter()
        result = func(*args, **kwds)
        end = time.perf_counter()
        duration = end - start
        call_str = _common.call_to_str(func, *args, **kwds)
        print(f'{call_str}: {duration} sec')
        return result
    return wrapped
