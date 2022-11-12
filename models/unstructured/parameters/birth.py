'''Birth rates.'''

import dataclasses

from ...parameters import birth


@dataclasses.dataclass
class _Birth:
    '''Common birth rate.'''

    @property
    def birth_rate_mean(self):
        '''`birth_rate_mean = death_rate`.'''
        return self.death_rate


@dataclasses.dataclass
class Constant(_Birth,
               birth.Constant):
    '''Constant birth rate.'''


@dataclasses.dataclass
class Periodic(_Birth,
               birth.Periodic):
    '''Periodic birth rate.'''
