'''Utilities.'''

import numpy


def arange(start, stop, step, endpoint=True, dtype=None):
    '''Like `numpy.arange()` but, if `endpoint` is True, ensure that
    `stop` is the output.'''
    arr = numpy.arange(start, stop, step, dtype=dtype)
    if endpoint and arr[-1] != stop:
        arr = numpy.hstack((arr, stop))
    return arr


def interp(x, xp, fp):
    '''Use `numpy.interp()` on each component of the array `fp`.'''
    fp = numpy.asarray(fp)
    if fp.ndim == 0:
        raise ValueError('Dimension of `fp` must be at least 1!')
    elif fp.ndim == 1:
        f = numpy.interp(x, xp, fp)
    else:
        f = numpy.stack([interp(x, xp, fp[..., j])
                         for j in range(fp.shape[-1])],
                        axis=-1)
    return f
