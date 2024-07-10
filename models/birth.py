'''Birth.'''

import abc
import functools

import numpy


class _Birth(metaclass=abc.ABCMeta):
    '''Base for births.'''

    @abc.abstractmethod
    def rate(self, t):
        '''Constant birth rate.'''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def rate_max(self):
        '''Birth rate maximum.'''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def rate_min(self):
        '''Birth rate minimum.'''
        raise NotImplementedError

    def __init__(self, parameters):
        self.variation = parameters.birth_variation
        assert self.variation >= 0
        self.age_menarche = parameters.birth_age_menarche
        self.age_menopause = parameters.birth_age_menopause
        assert 0 <= self.age_menarche <= self.age_menopause
        # The mean will be set in
        # `models.parameters._ModelParameters()` to give 0 population
        # growth rate but it is deferred until the population model is
        # available that also includes death.
        self.mean = None

    def maternity(self, age):
        '''Maternity.'''
        # 1 between menarche and menopause,
        # 0 otherwise.
        return numpy.where(((self.age_menarche <= age)
                            & (age < self.age_menopause)),
                           1, 0)

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

    @functools.cached_property
    def rate_max(self):
        '''Birth rate maximum.'''
        return self.mean

    @functools.cached_property
    def rate_min(self):
        '''Birth rate minimum.'''
        return self.mean


class _BirthPeriodic(_Birth):
    '''Base for periodic birth rates.'''

    def __init__(self, parameters):
        super().__init__(parameters)
        assert self.variation > 0
        self.period = parameters.birth_period
        assert self.period > 0


class BirthSinusoidal(_BirthPeriodic):
    '''Sinusoidal birth rate.'''

    @functools.cached_property
    def _amplitude(self):
        return self.variation * numpy.sqrt(2)

    def rate(self, t):
        '''Periodic birth rate.'''
        theta = 2 * numpy.pi * t / self.period
        return self.mean * (1 + self._amplitude * numpy.cos(theta))

    @functools.cached_property
    def rate_max(self):
        '''Birth rate maximum.'''
        return self.mean * (1 + self._amplitude)

    @functools.cached_property
    def rate_min(self):
        '''Birth rate minimum.'''
        return self.mean * (1 - self._amplitude)


class BirthPeriodicPiecewiseLinear(_BirthPeriodic):
    '''Piecewise-linear birth rate.'''

    @functools.cached_property
    def _rate_max(self):
        if self.variation < 1 / numpy.sqrt(3):
            rate_max = 1 + numpy.sqrt(3) * self.variation
        else:
            rate_max = 3 / 2 * (1 + self.variation ** 2)
        return rate_max

    @functools.cached_property
    def _amplitude(self):
        if self.variation < 1 / numpy.sqrt(3):
            amplitude = (2 * numpy.sqrt(3) * self.variation
                         / (1 + numpy.sqrt(3) * self.variation))
        else:
            amplitude = 3 / 4 * (1 + self.variation ** 2)
        return amplitude

    def rate(self, t):
        '''Periodic birth rate.'''
        t_frac = numpy.mod(t, self.period)
        val = (
            self._rate_max
            * (1 + self._amplitude * (numpy.abs(1 - 2 * t_frac) - 1))
        )
        return self.mean * numpy.clip(val, 0, None)

    @functools.cached_property
    def rate_max(self):
        '''Birth rate maximum.'''
        val_max = self._rate_max
        return self.mean * val_max

    @functools.cached_property
    def rate_min(self):
        '''Birth rate minimum.'''
        val_min = self._rate_max * (1 - self._amplitude)
        return self.mean * numpy.clip(val_min, 0, None)


BirthPeriodic = BirthSinusoid


def Birth(parameters):
    '''Factory function for birth.'''
    if parameters.birth_variation == 0:
        return BirthConstant(parameters)
    else:
        return BirthPeriodic(parameters)
