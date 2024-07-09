'''Numerical utilities.'''

import tempfile
import warnings

import numpy
import pandas

from . import sparse as sparse_


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
    width = stop - start
    # See if using `n_steps = floor(width / step)` is close enough to
    # `stop`.
    n_steps = numpy.floor(width / step)
    stop_new = start + n_steps * step
    assert stop_new <= stop
    if not numpy.isclose(stop_new, stop):
        # Add another step.
        stop_new += step
        assert stop_new >= stop
    return arange(start, stop_new, step)


def is_increasing(arr):
    '''Check whether `arr` is increasing.'''
    return numpy.all(numpy.diff(arr) > 0)


def check_nonnegative(y):
    '''Check that `y` is non-negative.'''
    if not numpy.all((y >= 0) | numpy.isclose(y, 0)):
        warnings.warn('Some entries are negative.', stacklevel=2)


def rate_make_finite(rates):
    '''Set any positive infinity values in `rates` to the closest
    previous finite value.'''
    if numpy.ndim(rates) != 1:
        raise NotImplementedError
    rates = pandas.Series(rates)
    rates[numpy.isposinf(rates)] = numpy.NaN
    rates = rates.ffill() \
                 .to_numpy()
    assert numpy.isfinite(rates).all()
    return rates


def weighted_sum(y, weights):
    '''`(y * weights).sum()`'''
    return (y * weights).sum()


def identity(*args, sparse=False, **kwds):
    '''Make an identity matrix.'''
    if sparse:
        eye = sparse_.identity(*args, **kwds)
    else:
        eye = numpy.identity(*args, **kwds)
    return eye


def memmaptemp(**kwds):
    '''Create an array memory-mapped to a temporary file.'''
    file_ = tempfile.TemporaryFile()
    return numpy.memmap(file_, mode='w+', **kwds)
