'''Model base class.'''

from . import birth
from . import parameters


class Base:
    '''Base class for models.'''

    states = ('maternal_immunity', 'susceptible', 'exposed',
              'infectious', 'recovered')

    def __init__(self, **kwds):
        self.parameters = parameters.Parameters(**kwds)
        self.birth_rate = birth.Rate(self.parameters)
