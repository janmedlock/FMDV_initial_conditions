'''Model parameters.'''

import dataclasses

import numpy

from . import birth, death, progression, recovery, transmission, waning


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


class _MixinBase:
    '''Mixin to initialize parameters for models.'''

    def _init_parameters(self, **kwds):
        '''Initialize model parameters.'''
        parameters = Parameters(**kwds)
        self.death = death.Death(parameters)
        self.birth = birth.Birth(parameters, self.death)
        self.progression = progression.Progression(parameters)
        self.recovery = recovery.Recovery(parameters)
        self.transmission = transmission.Transmission(parameters)
        self.waning = waning.Waning(parameters)


class AgeIndependent(_MixinBase):
    '''Mixin to initialize parameters for age-independent models.'''

    def _init_parameters(self, **kwds):
        '''Initialize model parameters.'''
        super()._init_parameters(**kwds)
        # Use `self.birth` with age-dependent `.mean` to find
        # `self.death_rate_mean`.
        self.death_rate_mean = self.death.rate_population_mean(self.birth)
        # Set `self.birth.mean` so this age-independent model has
        # zero population growth rate. For this model, the mean
        # population growth rate is
        # `self.birth.mean - self.death_rate_mean`.
        self.birth.mean = self.death_rate_mean


class AgeDependent(_MixinBase):
    '''Mixin to initialize parameters for age-dependent models.'''
