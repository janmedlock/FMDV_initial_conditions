'''Transformations.'''

import numpy
import scipy.special

from . import numerical


class Identity:
    '''`y` -> `y`.'''

    @staticmethod
    def __call__(y):
        x = y
        return x

    @staticmethod
    def inverse(x):
        y = x
        return y


class Logarithm:
    '''`y` -> `log(y - a)`.'''

    def __init__(self, a=0, weights=1):
        self.a = a
        self.weights = weights

    def __call__(self, y):
        z = y - self.a
        x = numpy.log(z * self.weights)
        return x

    def inverse(self, x):
        z = numpy.exp(x) / self.weights
        y = z + self.a
        return y


class Logit:
    '''`y` -> `logit((y - a) / (b - a))`.'''

    def __init__(self, a=0, b=1, weights=1):
        self.a = a
        self.b = b
        self.weights = weights

    def __call__(self, y):
        z = (y - self.a) / (self.b - self.a)
        x = scipy.special.logit(z * self.weights)
        return x

    def inverse(self, x):
        z = scipy.special.expit(x) / self.weights
        y = z * (self.b - self.a) + self.a
        return y


class Simplex:
    '''Use simplex coordinates.'''

    def __init__(self, weights=1):
        self.weights = weights

    @staticmethod
    def _w_mid(size):
        # The w values that map to x[k] = 0.
        return 1 / numpy.arange(size, 1, -1)

    def __call__(self, y):
        size = len(y)
        # `z` is on the unit simplex.
        z = (y * self.weights) / numerical.weighted_sum(y, self.weights)
        remainder = 1 - numpy.hstack([0, numpy.cumsum(z[:-1])])
        assert (remainder >= 0).all()
        w_mid = self._w_mid(size)
        # w = z[:-1] / remainder[:-1]
        # but if z[k] = remainder[k] = 0,
        # use w[k] = w_mid[k] so that x[k] = 0.
        num = z[:-1]
        den = remainder[:-1]
        zdz = (num == 0) & (den == 0)
        num[zdz] = w_mid[zdz]
        den[zdz] = 1
        w = num / den
        x = scipy.special.logit(w) - scipy.special.logit(w_mid)
        return x

    def inverse(self, x):
        size = len(x) + 1
        w_mid = self._w_mid(size)
        w = scipy.special.expit(x + scipy.special.logit(w_mid))
        z = numpy.empty(size)
        remainder = 1
        for k in range(size - 1):
            z[k] = remainder * w[k]
            remainder -= z[k]
        z[-1] = remainder
        y = z / self.weights
        return y
