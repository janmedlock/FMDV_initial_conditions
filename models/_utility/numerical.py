'''Numerical utilities.'''

import numpy
import pandas


def arange(start, stop, step, endpoint=True, dtype=None):
    '''Like `numpy.arange()` but, if `endpoint` is True, ensure that
    `stop` is the output.'''
    arr = numpy.arange(start, stop, step, dtype=dtype)
    if endpoint and arr[-1] != stop:
        arr = numpy.hstack((arr, stop))
    return arr


def build_t(start, stop, step):
    '''Increase `stop` if needed so that the interval is divided into
    a whole number of steps of width `step`.'''
    stop = start + numpy.ceil((stop - start) / step) * step
    return arange(start, stop, step)


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
    assert numpy.all((y >= 0) | numpy.isclose(y, 0))


def rate_make_finite(rates):
    '''Set any positive infinity values in `rates` to the closest
    previous finite value.'''
    if numpy.ndim(rates) != 1:
        raise NotImplementedError
    rates = pandas.Series(rates)
    rates[numpy.isposinf(rates)] = numpy.NaN
    rates = rates.fillna(method='ffill') \
                 .to_numpy()
    assert numpy.isfinite(rates).all()
    return rates


def weighted_sum(y, weights):
    '''`(y * weights).sum()`'''
    return (y * weights).sum()
