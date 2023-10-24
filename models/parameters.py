'''Model parameters.'''

import collections.abc
import dataclasses

import numpy

from . import birth, death, progression, recovery, transmission, waning
from .age_structured import _population


class _MappingMixin(collections.abc.Mapping):
    '''Mix-in methods that allows `_Parameters()` instances to be used
    as keyword mappings in function calls like `f(**parameters)`.
    `parameters['attr']` returns `parameters.attr`.  This mapping
    interface is *not mutable*.'''

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield field.name

    def __len__(self):
        return len(dataclasses.fields(self))


@dataclasses.dataclass
class _Parameters(_MappingMixin):
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


class _PeriodMixIn:
    '''Mixin for the period of the parameters.'''

    @property
    def period(self):
        '''The only periodic parameter is birth.'''
        return self.birth.period


class _ModelParametersPopulation(_PeriodMixIn):
    '''Model parameters for population models. For efficient caching,
    the infection parameters are dropped.'''

    def __init__(self, model_parameters):
        self.birth = model_parameters.birth
        self.death = model_parameters.death


class _ModelParameters(_PeriodMixIn):
    '''Base model parameters.'''

    def __init__(self, **kwds):
        parameters = Parameters(**kwds)
        # Setup the population parameters.
        self.birth = birth.Birth(parameters)
        self.death = death.Death(parameters)
        self._population_model = _population.Model(parameters=self)
        self._set_for_zero_population_growth()
        # Setup the infection parameters.
        self.progression = progression.Progression(parameters)
        self.recovery = recovery.Recovery(parameters)
        self.transmission = transmission.Transmission(parameters)
        self.waning = waning.Waning(parameters)

    def _set_for_zero_population_growth(self, **kwds):
        '''Set the mean birth and death rates so that the population
        growth rate is zero.'''
        # For the age-dependent parameters, set the mean birth rate
        # for zero population growth and do not adjust death.
        if self.birth.mean is None:
            self.birth.mean = 0.5  # Starting guess.
        bscl = self._population_model.birth_scaling_for_zero_population_growth(
            **kwds
        )
        self.birth.mean *= bscl
        assert self.birth.mean > 0


class ModelParametersAgeDependent(_ModelParameters):
    '''Model parameters for age-dependent models.'''

    def _stable_age_density(self, **kwds):
        '''Get the stable age density.'''
        return self._population_model.stable_age_density(**kwds)


class ModelParametersAgeIndependent(_ModelParameters):
    '''Model parameters for age-independent models.'''

    def _set_for_zero_population_growth(self, **kwds):
        '''Set the mean birth and death rates so that the population
        growth rate is zero.'''
        # Set the age-dependent mean birth and death rates.
        super()._set_for_zero_population_growth(**kwds)
        # Find the mean death rate over age using the stable age
        # density of the age-dependent population model with the
        # age-dependent birth and death rates.
        self.death_rate_mean = self.death.rate_population_mean(
            self._population_model
        )
        assert self.death_rate_mean > 0
        # The population growth rate for the age-independent model is
        # `self.birth.mean - self.death_rate_mean`, so set the mean birth
        # rate to `self.death_rate_mean` so that the population growth
        # rate is 0.
        self.birth.mean = self.death_rate_mean
        assert self.birth.mean > 0
