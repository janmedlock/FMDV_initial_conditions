'''Waiting times.'''

import numpy
import scipy.stats


class Gamma:
    '''A gamma-distributed waiting time.'''

    def __init__(self, mean, shape):
        self.mean = mean
        assert self.mean > 0
        self.shape = shape
        assert self.shape > 0
        scale = self.mean / self.shape
        self._rv = scipy.stats.gamma(self.shape, scale=scale)

    def rate(self, z):
        '''Hazard.'''
        # `self._rv.pdf(z) / self._rv.sf(z)`, but using logs for
        # better over- & under-flow.
        loghazard = self._rv.logpdf(z) - self._rv.logsf(z)
        return numpy.exp(loghazard)

    def survival(self, z):
        '''Survival function.'''
        return self._rv.sf(z)
