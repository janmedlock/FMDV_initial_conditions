'''Recovery from infectious to recovered.'''

import scipy.stats


class Recovery:
    '''Recovery from infectious to recovered.'''

    def __init__(self, parameters):
        self.mean = parameters.recovery_mean
        self.shape = parameters.recovery_shape
        scale = self.mean / self.shape
        self._rv = scipy.stats.gamma(self.shape, scale=scale)

    def rate(self, time_since_entry):
        '''Recovery rate.'''
        return self._rv.hazard(time_since_entry)
