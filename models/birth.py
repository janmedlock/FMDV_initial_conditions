'''Birth.'''

import numpy

from .age_structured import _population


class _Birth:
    '''Base for births.'''

    def __init__(self, parameters, death, *args, display=False, **kwds):
        self.variation = parameters.birth_variation
        assert self.variation >= 0
        self.period = parameters.birth_period
        assert self.period >= 0
        self.age_menarche = parameters.birth_age_menarche
        self.age_menopause = parameters.birth_age_menopause
        assert 0 <= self.age_menarche <= self.age_menopause
        self._death = death
        # `self.mean` must be set to initialize `_population.Model()`
        # and to call
        # `_population.Model.birth_scaling_for_zero_population_growth()`
        # in `self._mean_for_zero_population_growth()`, so set a
        # starting guess for `self.mean`.
        self.mean = 0.5
        self._model = _population.Model(self, self._death, *args, **kwds)
        self.mean = self._mean_for_zero_population_growth(display=display)
        assert self.mean >= 0

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

    def _mean_for_zero_population_growth(self, **kwds):
        '''Get the value for `self.mean` that gives zero population
        growth rate.'''
        scale = self._model.birth_scaling_for_zero_population_growth(**kwds)
        mean_for_zero_population_growth = scale * self.mean
        return mean_for_zero_population_growth

    def _integral_over_a(self, arr, *args, **kwds):
        '''Integrate `arr` over age.'''
        return self._model.integral_over_a(arr, *args, **kwds)

    def _stable_age_density(self, **kwds):
        '''Get the stable age density.'''
        return self._model.stable_age_density(**kwds)

    def _age_max(self):
        '''Get the last age where `.maternity()` changes.'''
        if numpy.isfinite(self.age_menopause):
            age_max = self.age_menopause
        else:
            age_max = self.age_menarche
        return age_max


class BirthConstant(_Birth):
    '''Constant birth rate.'''

    def __init__(self, parameters, death):
        super().__init__(parameters, death)
        assert self.variation == 0

    # `_population.Model.birth_scaling_for_zero_population_growth()`
    # has a shortcut when `period = 0`, so always return that value.
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

    def __init__(self, parameters, death):
        super().__init__(parameters, death)
        assert self.variation > 0

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
