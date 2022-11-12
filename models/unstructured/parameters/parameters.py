'''Parameters for unstructured models.'''

import dataclasses

from . import birth
from . import death
from ... import parameters


@dataclasses.dataclass
class ParametersBirthPeriodic(parameters.Parameters,
                              birth.Periodic,
                              death.BirthPeriodic):
    '''Parameters for unstructured models with periodic birth rate.'''


@dataclasses.dataclass
class ParametersBirthConstant(parameters.Parameters,
                              birth.Constant,
                              death.BirthConstant):
    '''Parameters for unstructured models with constant birth rate.'''
