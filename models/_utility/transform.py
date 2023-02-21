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


class Simplex:
    '''Transform to a simplex with sum y_j = `scale`.'''

    def __init__(self, scale=1, shift=1e-8, weights=1):
        self.scale = scale
        self.shift = shift
        self.weights = numpy.asarray(weights)

    @staticmethod
    def _w_mid(K):
        # The w values that map to x[k] = 0.
        return 1 / numpy.arange(K, 1, -1)

    def __call__(self, y):
        K = len(y)
        z = (numpy.asarray(y) + self.shift) * self.weights
        assert (z >= 0).all()
        z /= z.sum()
        remainder = 1 - numpy.hstack([0, numpy.cumsum(z[:-1])])
        assert (remainder >= 0).all()
        w_mid = self._w_mid(K)
        # w = z[:-1] / remainder[:-1]
        # but if z[k] = remainder[k] = 0,
        # use w[k] = w_mid[k] so that x[k] = 0.
        w = numpy.ma.divide(z[:-1], remainder[:-1]) \
                    .filled(w_mid)
        x = scipy.special.logit(w) - scipy.special.logit(w_mid)
        return x

    def inverse(self, x):
        K = len(x) + 1
        w_mid = self._w_mid(K)
        w = scipy.special.expit(x + scipy.special.logit(w_mid))
        z = numpy.empty(K)
        remainder = 1
        for k in range(K - 1):
            z[k] = remainder * w[k]
            remainder -= z[k]
        z[-1] = remainder
        assert (z >= 0).all()
        y = numpy.clip(((self.scale * z - self.shift)
                        / self.weights),
                       0, None)
        y *= self.scale / y.sum()
        return y
