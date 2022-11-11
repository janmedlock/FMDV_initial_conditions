'''Parameters for unstructured models.'''

import dataclasses

from . import birth
from ... import parameters


@dataclasses.dataclass
class Parameters(parameters.Parameters):
    '''Parameters for unstructured models.'''

    death_rate: float = 0.1  # per year


@dataclasses.dataclass
class ParametersBirthPeriodic(Parameters,
                              birth.Periodic):
    '''Parameters for unstructured models with periodic birth rate.'''


@dataclasses.dataclass
class ParametersBirthConstant(Parameters,
                              birth.Constant):
    '''Parameters for unstructured models with constant birth rate.'''
