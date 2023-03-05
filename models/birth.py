'''Birth.'''

import numpy


class _Birth:
    '''Base for births.'''

    def __init__(self, parameters, *args, **kwds):
        self.variation = parameters.birth_variation
        assert self.variation >= 0
        self.period = parameters.birth_period
        assert self.period >= 0
        self.age_menarche = parameters.birth_age_menarche
        self.age_menopause = parameters.birth_age_menopause
        assert 0 <= self.age_menarche <= self.age_menopause
        # Set a starting guess for `self.mean`, which will be replaced
        # in `_init_post()`.
        self.mean = 0.5

    def _mean_for_zero_population_growth(self, population, **kwds):
        '''Get the value for `self.mean` that gives zero population
        growth rate.'''
        scale = population.birth_scaling_for_zero_population_growth(**kwds)
        mean_for_zero_population_growth = scale * self.mean
        return mean_for_zero_population_growth

    def _init_post(self, population, **kwds):
        '''Final initialization.'''
        self.mean = self._mean_for_zero_population_growth(population, **kwds)
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

    def _age_max(self):
        '''Get the last age where `.maternity()` changes.'''
        if numpy.isfinite(self.age_menopause):
            age_max = self.age_menopause
        else:
            age_max = self.age_menarche
        return age_max


class BirthConstant(_Birth):
    '''Constant birth rate.'''

    def __init__(self, parameters):
        super().__init__(parameters)
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

    def __init__(self, parameters):
        super().__init__(parameters)
        assert self.variation > 0

    def rate(self, t):
        '''Periodic birth rate.'''
        theta = 2 * numpy.pi * t / self.period
        return self.mean * (1 + self.amplitude * numpy.cos(theta))


def Birth(parameters):
    '''Factory function for birth.'''
    if parameters.birth_variation == 0:
        return BirthConstant(parameters)
    else:
        return BirthPeriodic(parameters)
