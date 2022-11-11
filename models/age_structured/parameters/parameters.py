'''Parameters for age-structured models.'''

import dataclasses

from . import birth
from . import death
from . import maternity
from ... import parameters


@dataclasses.dataclass
class Parameters(parameters.Parameters,
                 death.Death,
                 maternity.Maternity):
    '''Parameters for age-structured models.'''


@dataclasses.dataclass
class ParametersBirthPeriodic(Parameters,
                              birth.Periodic):
    '''Parameters for age-structured models with periodic birth rate.'''


@dataclasses.dataclass
class ParametersBirthConstant(Parameters,
                              birth.Constant):
    '''Parameters for age-structured models with constant birth rate.'''
