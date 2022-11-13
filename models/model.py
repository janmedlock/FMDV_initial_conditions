'''Model base class.'''

import numpy

from . import birth
from . import death
from . import parameters
from . import progression
from . import recovery
from . import transmission
from . import waning


class Base:
    '''Base class for models.'''

    states = ('maternal_immunity', 'susceptible', 'exposed',
              'infectious', 'recovered')

    # This determines whether offspring are born with maternal
    # immunity.
    states_with_antibodies = ['recovered']

    def __init__(self, **kwds):
        self.parameters = parameters.Parameters(**kwds)
        self.death = death.Death(self.parameters)
        self.birth = birth.Birth(self.parameters,
                                 self.death)
        self.progression = progression.Progression(self.parameters)
        self.recovery = recovery.Recovery(self.parameters)
        self.transmission = transmission.Transmission(self.parameters)
        self.waning = waning.Waning(self.parameters)
        self._states_have_antibodies = self._get_states_have_antibodies()

    def _get_states_have_antibodies(self):
        '''True or False for whether each state has antibodies.'''
        return numpy.isin(self.states, self.states_with_antibodies)
