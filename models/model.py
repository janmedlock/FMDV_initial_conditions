'''Model base class.'''

from . import birth
from . import death
from . import maternity
from . import parameters


class Base:
    '''Base class for models.'''

    states = ('maternal_immunity', 'susceptible', 'exposed',
              'infectious', 'recovered')

    def __init__(self, **kwds):
        self.parameters = parameters.Parameters(**kwds)
        self.death_rate = death.DeathRate(self.parameters)
        self.maternity_rate = maternity.MaternityRate(self.parameters)
        self.birth_rate = birth.BirthRate(self.parameters,
                                          self.death_rate,
                                          self.maternity_rate)
