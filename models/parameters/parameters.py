'''Parameters common to all models.'''

import dataclasses

from . import birth
from . import death
from . import maternal_immunity
from . import maternity
from . import progression
from . import recovery
from . import transmission


@dataclasses.dataclass
class Parameters(death.Death,
                 maternal_immunity.Waning,
                 maternity.Maternity,
                 progression.Progression,
                 recovery.Recovery,
                 transmission.Transmission):
    '''Parameters for all models.'''


@dataclasses.dataclass
class ParametersBirthPeriodic(Parameters,
                              birth.Periodic):
    '''Parameters for all models with periodic birth rate.'''


@dataclasses.dataclass
class ParametersBirthConstant(Parameters,
                              birth.Constant):
    '''Parameters for all models with constant birth rate.'''
