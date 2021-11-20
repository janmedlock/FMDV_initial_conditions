'''Utilities.'''

import math

import numpy


def arange(start, stop, step, endpoint=True, dtype=None):
    '''Like `numpy.arange()` but ensure that `stop - step` is an integer
    multiple of `step`, which may increase `stop`, and last point is
    in the output if `endpoint` is True.'''
    # Get the smallest integer `num` that is at least as large as
    # `(stop - start) / step`.
    num = math.ceil((stop - start) / step)
    # Make `stop - start` an integer multiple of `step`.
    # This may increase the value of `stop`.
    stop = start + num * step
    if endpoint:
        num += 1
    return numpy.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype)


def interp(x, xp, fp):
    '''Use `numpy.interp()` on each component of the array `fp`.'''
    fp = numpy.asarray(fp)
    if fp.ndim == 0:
        raise ValueError('Dimension of `fp` must be at least 1.')
    elif fp.ndim == 1:
        return numpy.interp(x, xp, fp)
    else:
        return numpy.stack([interp(x, xp, fp[..., j])
                            for j in range(fp.shape[-1])],
                           axis=-1)
