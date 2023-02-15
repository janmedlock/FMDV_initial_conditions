'''Recovery from infectious to recovered.'''

import scipy.stats

from . import _utility


class Recovery:
    '''Recovery from infectious to recovered.'''

    def __init__(self, parameters):
        self.mean = parameters.recovery_mean
        self.shape = parameters.recovery_shape
        scale = self.mean / self.shape
        self._rv = scipy.stats.gamma(self.shape, scale=scale)

    def rate(self, time_since_entry):
        '''Recovery rate.'''
        return _utility.hazard(self._rv, time_since_entry)
