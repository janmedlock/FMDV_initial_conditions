'''Progression for exposed to infectious.'''

import scipy.stats


class Progression:
    '''Progression for exposed to infectious.'''

    def __init__(self, parameters):
        self.mean = parameters.progression_mean
        self.shape = parameters.progression_shape
        scale = self.mean / self.shape
        self._rv = scipy.stats.gamma(self.shape, scale=scale)

    def rate(self, time_since_entry):
        '''Progression rate.'''
        return self._rv.hazard(time_since_entry)
