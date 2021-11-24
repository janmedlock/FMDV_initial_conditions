'''Utilities.'''

import numpy


def all_subclasses(cls):
    '''Find all subclasses, subsubclasses, etc. of `cls`.'''
    for c in cls.__subclasses__():
        yield c
        for s in all_subclasses(c):
            yield s


def arange(start, stop, step, endpoint=True, dtype=None):
    '''Like `numpy.arange()` but, if `endpoint` is True, ensure that
    `stop` is the output.'''
    arr = numpy.arange(start, stop, step, dtype=dtype)
    if endpoint and arr[-1] != stop:
        arr = numpy.hstack((arr, stop))
    return arr


def sort_by_real_part(arr):
    '''Sort the elements of `arr` by real part.'''
    order = arr.real.argsort()
    return arr[order]


class TransformConstantSum:
    '''Reduce the dimension of `y` by 1 using its sum.'''

    def __init__(self, y):
        self.y_sum = y.sum()

    @staticmethod
    def __call__(y):
        '''Reduce the dimension of `y`.'''
        return y[:-1]

    def inverse(self, x):
        '''Expand the dimension of `x`.'''
        return numpy.hstack((x, self.y_sum - x.sum()))
