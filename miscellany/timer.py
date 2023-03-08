'''Utilities for timing code.'''

import functools
import time


def _call_str(func, *args, **kwds):
    '''Build a string of the function call.'''
    args_kwds_strs = []
    if len(args) > 0:
        args_str = ', '.join(map(str, args))
        args_kwds_strs.append(args_str)
    if len(kwds) > 0:
        kwds_str = ', '.join(f'{k}={v}' for (k, v) in kwds.items())
        args_kwds_strs.append(kwds_str)
    args_kwds_str = ', '.join(args_kwds_strs)
    call_str = f'{func.__name__}({args_kwds_str})'
    return call_str


def timer(func):
    '''A decorator to time function calls.'''
    @functools.wraps(func)
    def wrapped(*args, **kwds):
        t0 = time.perf_counter()
        result = func(*args, **kwds)
        t1 = time.perf_counter()
        t = t1 - t0
        call_str = _call_str(func, *args, **kwds)
        print(f'{call_str}: {t} sec')
        return result
    return wrapped
