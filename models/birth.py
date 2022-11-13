'''Birth.'''

import numpy

from . import _population


class _BirthRate:
    '''Base for birth rate.'''

    def __init__(self, parameters, death_rate, maternity_rate):
        self.variation = parameters.birth_variation
        self.period = parameters.birth_period
        self.mean = self._mean_for_zero_population_growth(death_rate,
                                                          maternity_rate)

    def _mean_for_zero_population_growth(self, death_rate, maternity_rate):
        '''Get the value for `self.mean` that gives zero population
        growth rate.'''
        # `self.mean` must be set for
        # `_population.birth_scaling_for_zero_population_growth()` to
        # work. If it wasn't set before, we'll set it to a starting
        # guess, and it will be unset after.
        if mean_unset := not hasattr(self, 'mean'):
            self.mean = 0.5  # Starting guess.
        scale = _population.birth_scaling_for_zero_population_growth(
            self, death_rate, maternity_rate)
        mean_for_zero_population_growth = scale * self.mean
        if mean_unset:
            del self.mean
        return mean_for_zero_population_growth


class BirthRateConstant(_BirthRate):
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
        '''Constant birth rate.'''
        return self.mean * numpy.ones_like(t)


class BirthRatePeriodic(_BirthRate):
    '''Periodic birth rate.'''

    def __call__(self, t):
        '''Periodic birth rate.'''
        amplitude = self.variation * numpy.sqrt(2)
        theta = 2 * numpy.pi * t / self.period
        return self.mean * (1 + amplitude * numpy.cos(theta))


def BirthRate(parameters, death_rate, maternity_rate):
    '''Factory function to build the birth rate.'''
    if parameters.birth_variation == 0:
        return BirthRateConstant(parameters, death_rate, maternity_rate)
    else:
        return BirthRatePeriodic(parameters, death_rate, maternity_rate)
