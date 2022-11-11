'''Birth rates.'''

import dataclasses
import functools

from . import _population_growth
from .. import death
from .. import maternity
from ....parameters import birth


@dataclasses.dataclass
class _Birth:
    '''Common birth rate.'''

    @functools.cached_property
    def birth_rate_mean(self):
        '''Set `birth_rate_mean` to the value that gives zero
        population growth rate.'''
        # Because of `functools.cached_property()`, when this starts,
        # there is no `self.birth_rate_mean` attribute.
        self.birth_rate_mean = 0.5  # Starting guess.
        scale = _population_growth.get_birth_scaling_for_no_pop_growth(
            self.death_rate, self.maternity_rate, self.birth_rate,
            self.birth_rate_period)
        self.birth_rate_mean *= scale
        # After this ends, `functools.cached_property()` sets
        # `self.birth_rate_mean` to the return value.
        return self.birth_rate_mean


@dataclasses.dataclass
class Constant(_Birth,
               birth.Constant):
    '''Constant birth rate.'''


@dataclasses.dataclass
class Periodic(_Birth,
               birth.Periodic):
    '''Periodic birth rate.'''
