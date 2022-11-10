'''Birth rate.'''

import numpy


class PeriodicBirthRateMixin:
    def birth_rate(self, t):
        '''Periodic birth rate.'''
        mean = self.birth_rate_mean
        amplitude = self.parameters.birth_rate_variation * numpy.sqrt(2)
        theta = 2 * numpy.pi * t / self.PERIOD
        return mean * (1 + amplitude * numpy.cos(theta))
