'''Birth.'''

import numpy


class _Birth:
    '''Base for births.'''

    def __init__(self, parameters):
        self.variation = parameters.birth_variation
        assert self.variation >= 0
        self.age_menarche = parameters.birth_age_menarche
        self.age_menopause = parameters.birth_age_menopause
        assert 0 <= self.age_menarche <= self.age_menopause
        # The mean will be set in
        # `models.parameters.ModelAgeDependentParameters()` to give 0
        # population growth rate but it is deferred until the
        # population model is available that also includes death.
        self.mean = None

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
        self.period = None

    def rate(self, t):
        '''Constant birth rate.'''
        return self.mean * numpy.ones_like(t)


class BirthPeriodic(_Birth):
    '''Periodic birth rate.'''

    def __init__(self, parameters):
        super().__init__(parameters)
        assert self.variation > 0
        self.period = parameters.birth_period
        assert self.period > 0

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
