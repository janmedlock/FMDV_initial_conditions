'''Birth.'''

import abc
import functools

import numpy


class _Birth(metaclass=abc.ABCMeta):
    '''Base for births.'''

    @property
    @abc.abstractmethod
    def shape(self):
        '''Shape name.'''
        raise NotImplementedError

    @abc.abstractmethod
    def _rate(self, t):
        '''Birth rate before scaling by `self.mean`.'''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _rate_max(self):
        '''Birth-rate maximum before scaling by `self.mean`.'''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _rate_min(self):
        '''Birth-rate minimum before scaling by `self.mean`.'''
        raise NotImplementedError

    def __init__(self, parameters):
        self.variation = parameters.birth_variation
        assert self.variation >= 0
        self.age_menarche = parameters.birth_age_menarche
        self.age_menopause = parameters.birth_age_menopause
        assert 0 <= self.age_menarche <= self.age_menopause
        assert self._rate_max > 0
        assert 0 <= self._rate_min <= self._rate_max
        # The mean will be set in
        # `models.parameters._ModelParameters()` to give 0 population
        # growth rate but it is deferred until the population model is
        # available that also includes death.
        self.mean = None

    def rate(self, t):
        '''Birth rate.'''
        return self._rate(t) * self.mean

    @property
    def rate_max(self):
        '''Birth-rate maximum.'''
        return self._rate_max * self.mean

    @property
    def rate_min(self):
        '''Birth-rate minimum.'''
        return self._rate_min * self.mean

    def maternity(self, age):
        '''Maternity.'''
        # 1 between menarche and menopause,
        # 0 otherwise.
        return numpy.where(((self.age_menarche <= age)
                            & (age < self.age_menopause)),
                           1, 0)

    @property
    def age_max(self):
        '''Get the last age where `.maternity()` changes.'''
        if numpy.isfinite(self.age_menopause):
            age_max = self.age_menopause
        else:
            age_max = self.age_menarche
        return age_max


class BirthConstant(_Birth):
    '''Constant birth rate.'''

    shape = 'constant'

    def __init__(self, parameters):
        self.period = None
        super().__init__(parameters)
        assert self.variation == 0

    def _rate(self, t):
        '''Birth rate before scaling by `self.mean`.'''
        return numpy.ones_like(t)

    # Birth-rate maximum before scaling by `self.mean`.
    _rate_max = 1

    # Birth-rate minimum before scaling by `self.mean`.
    _rate_min = 1


class _BirthPeriodic(_Birth):
    '''Base for periodic birth rates.'''

    @property
    @abc.abstractmethod
    def _amplitude(self):
        '''Birth-rate amplitude before scaling by `self.mean`.'''
        raise NotImplementedError

    def __init__(self, parameters):
        self.period = parameters.birth_period
        assert self.period > 0
        super().__init__(parameters)
        assert self.variation > 0
        assert self._rate_min < self._rate_max
        assert self._amplitude > 0


class BirthSinusoidal(_BirthPeriodic):
    '''Sinusoidal birth rate.'''

    shape = 'sinusoidal'

    @functools.cached_property
    def _amplitude(self):
        '''Birth-rate amplitude before scaling by `self.mean`.'''
        amplitude = self.variation * numpy.sqrt(2)
        assert amplitude <= 1
        return amplitude

    def _rate(self, t):
        '''Birth rate before scaling by `self.mean`.'''
        theta = 2 * numpy.pi * t / self.period
        return 1 + self._amplitude * numpy.cos(theta)

    @property
    def _rate_max(self):
        '''Birth-rate maximum before scaling by `self.mean`.'''
        return 1 + self._amplitude

    @property
    def _rate_min(self):
        '''Birth-rate minimum before scaling by `self.mean`.'''
        return 1 - self._amplitude


class BirthPiecewiseLinear(_BirthPeriodic):
    '''Piecewise-linear birth rate.'''

    shape = 'piecewise_linear'

    # This is the threshold above which the rate is 0 for some times.
    _variation_threshold = 1 / numpy.sqrt(3)

    @functools.cached_property
    def _amplitude(self):
        '''Birth-rate amplitude before scaling by `self.mean`.'''
        if self.variation < self._variation_threshold:
            return (2 * numpy.sqrt(3) * self.variation
                    / (1 + numpy.sqrt(3) * self.variation))
        return 3 / 4 * (1 + self.variation ** 2)

    @functools.cached_property
    def _rate_max(self):
        '''Birth-rate maximum before scaling by `self.mean`.'''
        if self.variation < self._variation_threshold:
            return 1 + numpy.sqrt(3) * self.variation
        return 3 / 2 * (1 + self.variation ** 2)

    def _rate(self, t):
        '''Birth rate before scaling by `self.mean`.'''
        t_frac = numpy.mod(t / self.period, 1)
        val = 1 + self._amplitude * (numpy.abs(1 - 2 * t_frac) - 1)
        return self._rate_max * numpy.clip(val, 0, None)

    @property
    def _rate_min(self):
        '''Birth-rate minimum before scaling by `self.mean`.'''
        val = 1 - self._amplitude
        return self._rate_max * numpy.clip(val, 0, None)


def _subclasses(cls):
    yield cls
    for sub in cls.__subclasses__():
        yield from _subclasses(sub)


def Birth(parameters):  # pylint: disable=invalid-name
    '''Factory function for birth.'''
    if parameters.birth_variation == 0:
        parameters.birth_shape = 'constant'
    elif parameters.birth_shape == 'constant':
        parameters.birth_variation = 0
    for cls in _subclasses(_Birth):
        if cls.shape == parameters.birth_shape:
            break
    else:
        raise ValueError(f'Unknown {parameters.birth_shape=}!')
    return cls(parameters)
