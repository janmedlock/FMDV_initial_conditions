'''Parameters for birth.'''

import abc
import dataclasses

import numpy


@dataclasses.dataclass
class _Birth(metaclass=abc.ABCMeta):
    '''Base for birth rates.'''

    @property
    @abc.abstractmethod
    def birth_rate_mean(self):
        '''The mean birth rate over time.'''

    @abc.abstractmethod
    def birth_rate(self, t):
        '''The birth rate as a function of time.'''


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
