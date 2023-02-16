'''Birth.'''

import numpy

from . import _population


class _Birth:
    '''Base for births.'''

    def __init__(self, parameters, death):
        self.variation = parameters.birth_variation
        self.period = parameters.birth_period
        self.age_menarche = parameters.birth_age_menarche
        self.age_menopause = parameters.birth_age_menopause
        self.mean = self._mean_for_zero_population_growth(death)

    def maternity(self, age):
        '''Maternity.'''
        # 1 between menarche and menopause,
        # 0 otherwise.
        return numpy.where(((self.age_menarche <= age)
                            & (age < self.age_menopause)),
                           1, 0)

    @property
    def amplitude(self):
        return self.variation * numpy.sqrt(2)

    @property
    def rate_min(self):
        '''Birth rate minimum.'''
        return self.mean * (1 - self.amplitude)

    @property
    def rate_max(self):
        '''Birth rate maximum.'''
        return self.mean * (1 + self.amplitude)

    def _mean_for_zero_population_growth(self, death):
        '''Get the value for `self.mean` that gives zero population
        growth rate.'''
        # `self.mean` must be set for
        # `_population.birth_scaling_for_zero_population_growth()` to
        # work. If it wasn't set before, we'll set it to a starting
        # guess, and it will be unset after.
        if mean_unset := not hasattr(self, 'mean'):
            self.mean = 0.5  # Starting guess.
        scale = _population.birth_scaling_for_zero_population_growth(self,
                                                                     death)
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
        theta = 2 * numpy.pi * t / self.period
        return self.mean * (1 + self.amplitude * numpy.cos(theta))


def Birth(parameters, death):
    '''Factory function for birth.'''
    if parameters.birth_variation == 0:
        return BirthConstant(parameters, death)
    else:
        return BirthPeriodic(parameters, death)
