'''Model parameters.'''

import dataclasses


@dataclasses.dataclass
class _Parameters:
    '''Model parameters common to all SATs.'''

    birth_variation: float = 0.613         # unitless
    birth_period: float = 1                # years
    maternal_immunity_mean: float = 0.37   # years
    maternal_immunity_shape: float = 1.19  # unitless


@dataclasses.dataclass
class ParametersSAT1(_Parameters):
    '''Model parameters for SAT1.'''

    progression_mean: float = 0.5 / 365   # years
    progression_shape: float = 1.2        # unitless
    recovery_mean: float = 5.7 / 365      # years
    recovery_shape: float = 11.8          # unitless
    transmission_rate: float = 2.8 * 365  # per year


@dataclasses.dataclass
class ParametersSAT2(_Parameters):
    '''Model parameters for SAT2.'''

    progression_mean: float = 1.3 / 365   # years
    progression_shape: float = 1.6        # unitless
    recovery_mean: float = 4.6 / 365      # years
    recovery_shape: float = 8.7           # unitless
    transmission_rate: float = 1.6 * 365  # per year


@dataclasses.dataclass
class ParametersSAT3(_Parameters):
    '''Model parameters for SAT3.'''

    progression_mean: float = 2.8 / 365   # years
    progression_shape: float = 1.6        # unitless
    recovery_mean: float = 4.2 / 365      # years
    recovery_shape: float = 11.8          # unitless
    transmission_rate: float = 1.2 * 365  # per year


def Parameters(SAT=1, **kwds):
    '''Factory function to build model parameters for the given SAT.'''
    klass_name = f'ParametersSAT{SAT}'
    try:
        klass = globals()[klass_name]
    except KeyError:
        raise ValueError(f'{SAT=}')
    return klass(**kwds)
