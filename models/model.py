'''Model base class.'''

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

    def __init__(self, **kwds):
        self.parameters = parameters.Parameters(**kwds)
        self.death = death.Death(self.parameters)
        self.birth = birth.Birth(self.parameters,
                                 self.death)
        self.progression = progression.Progression(self.parameters)
        self.recovery = recovery.Recovery(self.parameters)
        self.transmission = transmission.Transmission(self.parameters)
        self.waning = waning.Waning(self.parameters)
