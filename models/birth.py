'''Birth rate.'''

import numpy

from . import _population


class _Rate:
    '''Base for birth rate.'''

    def __init__(self, parameters):
        self.variation = parameters.birth_variation
        self.period = parameters.birth_period
        self.mean = self._mean_for_zero_population_growth()

    def _mean_for_zero_population_growth(self):
        '''Get the value for `self.mean` that gives zero population
        growth rate.'''
        # `self.mean` must be set in order for
        # `_population.birth_scaling_for_zero_population_growth()` to
        # work. If it wasn't set before, it will be unset after.
        if mean_unset := not hasattr(self, 'mean'):
            self.mean = 0.5  # Starting guess.
        scale = _population.birth_scaling_for_zero_population_growth(self)
        mean_for_zero_population_growth = scale * self.mean
        if mean_unset:
            del self.mean
        return mean_for_zero_population_growth


class RateConstant(_Rate):
    '''Constant birth rate.'''

    # `_population.birth_scaling_for_zero_population_growth()` has a
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
