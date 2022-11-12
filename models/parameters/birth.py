'''Parameters for birth.'''

import abc
import dataclasses

import numpy

from .. import _population


@dataclasses.dataclass
class _Birth(metaclass=abc.ABCMeta):
    '''Base for birth rate.'''

    @abc.abstractmethod
    def birth_rate(self, t):
        '''The birth rate as a function of time.'''

    def __post_init__(self):
        self._set_birth_rate_mean_to_zero_pop_growth()
        try:
            post_init = super().__post_init__
        except AttributeError:
            pass
        else:
            post_init()

    def _set_birth_rate_mean_to_zero_pop_growth(self):
        '''Set `birth_rate_mean` to the value that gives zero
        population growth rate.'''
        self.birth_rate_mean = 0.5  # Starting guess.
        scale = _population.get_birth_scaling_for_zero_pop_growth(
            self.death_rate, self.maternity_rate, self.birth_rate,
            self.birth_rate_period)
        self.birth_rate_mean *= scale


@dataclasses.dataclass
class Constant(_Birth):
    '''Constant birth rate.'''

    # These are *not* handled by `dataclasses.dataclass`.
    birth_rate_variation = 0.  # unitless
    birth_rate_period = 0.     # year

    def birth_rate(self, t):
        '''Constant birth rate.'''
        return self.birth_rate_mean * numpy.ones_like(t)


@dataclasses.dataclass
class Periodic(_Birth):
    '''Periodic birth rate.'''

    birth_rate_variation: float = 0.613  # unitless
    birth_rate_period: float = 1.        # year

    def birth_rate(self, t):
        '''Periodic birth rate.'''
        amplitude = self.birth_rate_variation * numpy.sqrt(2)
        theta = 2 * numpy.pi * t / self.birth_rate_period
        return (self.birth_rate_mean
                * (1 + amplitude * numpy.cos(theta)))
