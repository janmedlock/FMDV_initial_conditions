'''Transformations.'''

import numpy
import scipy.special


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
    '''Transform by appending sum weights_j y_j.'''

    def __init__(self, scale=1, weights=1):
        assert scale > 0
        self.scale = scale
        assert numpy.all(weights > 0)
        self.weights = numpy.asarray(weights)

    @staticmethod
    def __call__(y):
        x = y[:-1]
        return x

    def total(self, y):
        return (y * self.weights).sum()

    def inverse(self, x):
        try:
            weights_end = self.weights[-1]
        except IndexError:
            weights_end = self.weights
        y = numpy.hstack([x, 0])
        total = self.total(y)
        if self.scale >= total:
            y[-1] = (self.scale - total) / weights_end
        else:
            y *= self.scale / total
        return y

    @classmethod
    def from_y(cls, y, *args, **kwds):
        '''Use `y` to find `scale`.'''
        self = cls(*args, **kwds)
        self.scale = self.total(y)
        return self


class ConstantSumLogarithm:
    '''Transform by dropping the last element and ensuring sum
    weights_j y_j = `scale` and use the logarithm on the remaining
    elements.'''

    def __init__(self, scale=1, weights=1):
        assert scale > 0
        self.scale = scale
        assert numpy.all(weights > 0)
        self.weights = numpy.asarray(weights)

    def total(self, y):
        return (y * self.weights).sum()

    def __call__(self, y):
        y = numpy.asarray(y)
        x = numpy.log(y[:-1])
        return x

    def inverse(self, x):
        try:
            weights_end = self.weights[-1]
        except IndexError:
            weights_end = self.weights
        y = numpy.hstack([numpy.exp(x), 0])
        total = self.total(y)
        if self.scale >= total:
            y[-1] = (self.scale - total) / weights_end
        else:
            y *= self.scale / total
        return y

    @classmethod
    def from_y(cls, y, *args, **kwds):
        '''Use `y` to find `scale`.'''
        self = cls(*args, **kwds)
        self.scale = self.total(y)
        return self
