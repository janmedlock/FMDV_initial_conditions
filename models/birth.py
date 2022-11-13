'''Birth.'''

import numpy

from . import _population


class _Birth:
    '''Base for births.'''

    def __init__(self, parameters, death, maternity):
        self.variation = parameters.birth_variation
        self.period = parameters.birth_period
        self.mean = self._mean_for_zero_population_growth(death,
                                                          maternity)

    def _mean_for_zero_population_growth(self, death, maternity):
        '''Get the value for `self.mean` that gives zero population
        growth rate.'''
        # `self.mean` must be set for
        # `_population.birth_scaling_for_zero_population_growth()` to
        # work. If it wasn't set before, we'll set it to a starting
        # guess, and it will be unset after.
        if mean_unset := not hasattr(self, 'mean'):
            self.mean = 0.5  # Starting guess.
        scale = _population.birth_scaling_for_zero_population_growth(
            self, death, maternity)
        mean_for_zero_population_growth = scale * self.mean
        if mean_unset:
            del self.mean
        return mean_for_zero_population_growth


class BirthConstant(_Birth):
    '''Constant birth rate.'''

    # `_population.birth_scaling_for_zero_population_growth()` has a
    # shortcut when `period = 0`, so always return that value.
    @property
    def period(self):
        return 0

    @period.setter
    def period(self, val):
        pass

    def rate(self, t):
        '''Constant birth rate.'''
        return self.mean * numpy.ones_like(t)


class BirthPeriodic(_Birth):
    '''Periodic birth rate.'''

    def rate(self, t):
        '''Periodic birth rate.'''
        amplitude = self.variation * numpy.sqrt(2)
        theta = 2 * numpy.pi * t / self.period
        return self.mean * (1 + amplitude * numpy.cos(theta))


def Birth(parameters, death, maternity):
    '''Factory function for birth.'''
    if parameters.birth_variation == 0:
        return BirthConstant(parameters, death, maternity)
    else:
        return BirthPeriodic(parameters, death, maternity)
