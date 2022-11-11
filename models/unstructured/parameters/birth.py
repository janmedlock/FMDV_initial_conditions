'''Birth rates.'''

import dataclasses
import functools

from ...parameters import birth


@dataclasses.dataclass
class _Birth:
    '''Common birth rate.'''

    @functools.cached_property
    def birth_rate_mean(self):
        return self.death_rate


@dataclasses.dataclass
class Constant(_Birth,
               birth.Constant):
    '''Constant birth rate.'''


@dataclasses.dataclass
class Periodic(_Birth,
               birth.Periodic):
    '''Periodic birth rate.'''
