'''Death rate for unstructured models.'''

import abc
import dataclasses

import scipy.integrate

from ... import age_structured
from ... import parameters
from ...parameters import _population


@dataclasses.dataclass
class _Death(metaclass=abc.ABCMeta):
    '''Death rate for unstructured models.'''

    @property
    @abc.abstractmethod
    def _ParametersAgeStructured(self):
        '''Corresponding age-structured parameters class.'''

    def __post_init__(self):
        self._set_death_rate_to_mean_of_age_structured()

    def _set_death_rate_to_mean_of_age_structured(self):
        kws = dataclasses.asdict(self)
        parameters = self._ParametersAgeStructured(**kws)
        stable_age_density = _population.get_stable_age_density(
            parameters.death_rate,
            parameters.maternity_rate,
            parameters.birth_rate,
            self.birth_rate_period)
        ages = stable_age_density.index
        death_total = scipy.integrate.trapz(parameters.death_rate(ages)
                                            * stable_age_density,
                                            ages)
        density_total = scipy.integrate.trapz(stable_age_density,
                                              ages)
        self.death_rate = death_total / density_total


@dataclasses.dataclass
class BirthConstant(_Death):
    '''Death rate for unstructured models with constant birth rate.'''

    _ParametersAgeStructured \
        = age_structured.parameters.ParametersBirthConstant


@dataclasses.dataclass
class BirthPeriodic(_Death):
    '''Death rate for unstructured models with periodic birth rate.'''

    _ParametersAgeStructured \
        = age_structured.parameters.ParametersBirthPeriodic
