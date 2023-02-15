'''Waning of maternal immunity.'''

import scipy.stats

from . import _utility


class Waning:
    '''Waning of maternal immunity.'''

    def __init__(self, parameters):
        self.mean = parameters.maternal_immunity_mean
        self.shape = parameters.maternal_immunity_shape
        scale = self.mean / self.shape
        self._rv = scipy.stats.gamma(self.shape, scale=scale)

    def rate(self, time_since_entry):
        '''Waning rate.'''
        return _utility.hazard(self._rv, time_since_entry)
