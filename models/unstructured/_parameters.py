'''Parameters for unstructured models.'''

import abc
import dataclasses

import scipy.integrate

from .. import parameters
from .. import _population


@abc.abstractmethod
@dataclasses.dataclass
class _Parameters:
    '''Parameters for unstructured models.'''

    @abc.abstractmethod
    def birth_rate(self, t):
        '''The birth rate as a function of time.'''

    def __post_init__(self):
        self._set_death_rate_mean()
        self.birth_rate_mean = self.death_rate_mean
        try:
            post_init = super().__post_init__
        except AttributeError:
            pass
        else:
            post_init()

    def _stable_age_density(self):
        return _population.get_stable_age_density(self.death_rate,
                                                  self.maternity_rate,
                                                  self.birth_rate,
                                                  self.birth_rate_period)

    def _set_death_rate_mean(self):
        stable_age_density = self._stable_age_density()
        ages = stable_age_density.index
        death_total = scipy.integrate.trapz(self.death_rate(ages)
                                            * stable_age_density,
                                            ages)
        density_total = scipy.integrate.trapz(stable_age_density,
                                              ages)
        self.death_rate_mean = death_total / density_total


@dataclasses.dataclass
class ParametersBirthConstant(parameters.ParametersBirthConstant,
                              _Parameters):
    '''Parameters for unstructured models with constant birth rate.'''


@dataclasses.dataclass
class ParametersBirthPeriodic(parameters.ParametersBirthPeriodic,
                              _Parameters):
    '''Parameters for unstructured models with periodic birth rate.'''
