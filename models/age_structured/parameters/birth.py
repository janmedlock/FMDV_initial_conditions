'''Birth rates.'''

import dataclasses

from ... import parameters
from ...parameters import _population_growth


@dataclasses.dataclass
class _Birth:
    '''Common birth rate.'''

    # Set `birth_rate_mean` to a dummy value to satisfy
    # `abc.ABCMeta()`. The actual value gets set in `__post_init__()`.
    birth_rate_mean = None

    def __post_init__(self):
        self._set_birth_rate_mean_to_no_pop_growth()

    def _set_birth_rate_mean_to_no_pop_growth(self):
        '''Set `birth_rate_mean` to the value that gives zero
        population growth rate.'''
        self.birth_rate_mean = 0.5  # Starting guess.
        scale = _population_growth.get_birth_scaling_for_no_pop_growth(
            self.death_rate, self.maternity_rate, self.birth_rate,
            self.birth_rate_period)
        self.birth_rate_mean *= scale


@dataclasses.dataclass
class Constant(_Birth,
               parameters.birth.Constant):
    '''Constant birth rate.'''


@dataclasses.dataclass
class Periodic(_Birth,
               parameters.birth.Periodic):
    '''Periodic birth rate.'''
