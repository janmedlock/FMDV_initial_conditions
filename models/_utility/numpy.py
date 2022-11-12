'''Numerical utilities.'''

import numpy


def arange(start, stop, step, endpoint=True, dtype=None):
    '''Like `numpy.arange()` but, if `endpoint` is True, ensure that
    `stop` is the output.'''
    arr = numpy.arange(start, stop, step, dtype=dtype)
    if endpoint and arr[-1] != stop:
        arr = numpy.hstack((arr, stop))
    return arr


def _sort_by(arr, fcn):
    order = fcn(arr).argsort()
    return arr[order]


def sort_by_abs(arr):
    '''Sort the elements of `arr` by absolute value.'''
    return _sort_by(arr, numpy.abs)


def sort_by_real_part(arr):
    '''Sort the elements of `arr` by real part.'''
    return _sort_by(arr, numpy.real)


def assert_nonnegative(y):
    '''Check that `y` is non-negative.'''
    assert (y >= 0).all(axis=None)
