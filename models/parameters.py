'''Model parameters.'''

import dataclasses

import numpy

from . import birth, death, progression, recovery, transmission, waning
from .age_structured import _population


@dataclasses.dataclass
class _Parameters:
    '''Model parameters common to all SATs.'''

    # Not handled by `dataclasses.dataclass()`.
    birth_period = 1                         # years

    birth_variation: float = 0.613           # unitless
    birth_age_menarche: float = 4            # years
    birth_age_menopause: float = numpy.PINF  # years
    maternal_immunity_mean: float = 0.37     # years
    maternal_immunity_shape: float = 1.19    # unitless


@dataclasses.dataclass
class ParametersSAT1(_Parameters):
    '''Model parameters for SAT1.'''

    progression_mean: float = 0.5 / 365      # years
    progression_shape: float = 1.2           # unitless
    recovery_mean: float = 5.7 / 365         # years
    recovery_shape: float = 11.8             # unitless
    transmission_rate: float = 2.8 * 365     # per year


@dataclasses.dataclass
class ParametersSAT2(_Parameters):
    '''Model parameters for SAT2.'''

    progression_mean: float = 1.3 / 365      # years
    progression_shape: float = 1.6           # unitless
    recovery_mean: float = 4.6 / 365         # years
    recovery_shape: float = 8.7              # unitless
    transmission_rate: float = 1.6 * 365     # per year


@dataclasses.dataclass
class ParametersSAT3(_Parameters):
    '''Model parameters for SAT3.'''

    progression_mean: float = 2.8 / 365      # years
    progression_shape: float = 1.6           # unitless
    recovery_mean: float = 4.2 / 365         # years
    recovery_shape: float = 11.8             # unitless
    transmission_rate: float = 1.2 * 365     # per year


def Parameters(SAT=1, **kwds):
    '''Factory function to build model parameters for the given SAT.'''
    klass_name = f'ParametersSAT{SAT}'
    try:
        klass = globals()[klass_name]
    except KeyError:
        raise ValueError(f'{SAT=}')
    return klass(**kwds)


class _ModelParameters:
    '''Model parameters.'''

    def __init__(self, **kwds):
        parameters = Parameters(**kwds)
        self.death = death.Death(parameters)
        self.birth = birth.Birth(parameters)
        self.progression = progression.Progression(parameters)
        self.recovery = recovery.Recovery(parameters)
        self.transmission = transmission.Transmission(parameters)
        self.waning = waning.Waning(parameters)
        self._init_post()

    def _init_post(self, *args, **kwds):
        '''Final initialization.'''
        self._population = _population.Model(self.birth, self.death,
                                             *args, **kwds)
        self.birth._init_post(self._population)

    def _stable_age_density(self, **kwds):
        '''Get the stable age density.'''
        return self._population.stable_age_density(**kwds)


class ModelParametersAgeIndependent(_ModelParameters):
    '''Model parameters for age-independent models.'''

    def _death_rate_mean(self):
        '''Find the mean death rate.'''
        # Find the mean of the death rate over age using the stable
        # age density of the age-dependent population model with the
        # age-dependent birth and death rates. The result is the
        # death_rate for an age-independent model.
        death_rate_mean = self.death.rate_population_mean(self._population)
        return death_rate_mean

    def _birth_rate_mean_for_zero_population_growth(self):
        '''Get the birth rate mean that gives zero population growth.'''
        # For the age-independent population model, the mean over time
        # of the population growth rate is `self.birth.mean -
        # self.death_rate_mean`. Setting the mean of the birth rate to
        # `self.death_rate_mean()` ensures the age-independent
        # population model has zero mean population growth rate, just
        # like the mean population growth rate for the age-dependent
        # population model.
        birth_mean = self.death_rate_mean
        return birth_mean

    def _init_post(self):
        '''Final initialization.'''
        super()._init_post()
        self.death_rate_mean = self._death_rate_mean()
        self.birth.mean = self._birth_rate_mean_for_zero_population_growth()


class ModelParametersAgeDependent(_ModelParameters):
    '''Model parametets for age-dependent models.'''
