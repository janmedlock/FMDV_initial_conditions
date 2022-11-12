'''Birth rate.'''

import functools

import numpy

from . import death
from . import maternity
from . import _population


class _Rate:
    '''Base for birth rate.'''

    def __init__(self, parameters):
        self.variation = parameters.birth_variation
        self.period = parameters.birth_period
        self._set_mean_to_zero_pop_growth()

    def _set_mean_to_zero_pop_growth(self):
        '''Set `birth_rate_mean` to the value that gives zero
        population growth rate.'''
        self.mean = 0.5  # Starting guess.
        scale = _population.get_birth_scaling_for_zero_pop_growth(self)
        self.mean *= scale


class RateConstant(_Rate):
    '''Constant birth rate.'''

    # `_population.get_birth_scaling_for_zero_pop_growth()` has a
    # shortcut when `period = 0`, so always return that value.
    @property
    def period(self):
        return 0

    @period.setter
    def period(self, val):
        pass

    def __call__(self, t):
        return self.mean * numpy.ones_like(t)


class RatePeriodic(_Rate):
    '''Periodic birth rate.'''

    def __call__(self, t):
        amplitude = self.variation * numpy.sqrt(2)
        theta = 2 * numpy.pi * t / self.period
        return self.mean * (1 + amplitude * numpy.cos(theta))


def Rate(parameters):
    '''Factory function to build the birth rate.'''
    if parameters.birth_variation == 0:
        return RateConstant(parameters)
    else:
        return RatePeriodic(parameters)
