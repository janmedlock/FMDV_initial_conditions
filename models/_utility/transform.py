'''Transformations.'''

import numpy
import scipy.special

from . import numerical


class Identity:
    '''`y` -> `y`.'''

    @staticmethod
    def __call__(y):
        return y

    @staticmethod
    def inverse(x):
        return x


class Logarithm:
    '''`y` -> `log(y)`.'''

    @staticmethod
    def __call__(y):
        return numpy.log(y)

    @staticmethod
    def inverse(x):
        return numpy.exp(x)


class ConstantSum:
    '''Transform by dropping the last element and ensuring
    sum weights_j y_j = `scale`.'''

    def __init__(self, scale=1, weights=1):
        assert scale > 0
        self.scale = scale
        assert numpy.all(weights > 0)
        self.weights = numpy.asarray(weights)

    @classmethod
    def from_y(cls, y, weights=1):
        '''Use `y` to find `scale`.'''
        numerical.assert_nonnegative(y)
        scale = (y * weights).sum()
        return cls(scale=scale, weights=weights)

    def __call__(self, y):
        x = y[:-1]
        return x

    def inverse(self, x):
        try:
            weights_end = self.weights[-1]
        except IndexError:
            weights_end = self.weights
        y = numpy.hstack([x, 0])
        total = (y * self.weights).sum()
        if self.scale >= total:
            y[-1] = (self.scale - total) / weights_end
        else:
            y *= self.scale / total
        return y
