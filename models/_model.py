'''Model base class.'''

import abc


class Model(metaclass=abc.ABCMeta):
    '''Base class for models.'''

    states = ('maternal_immunity', 'susceptible', 'exposed',
              'infectious', 'recovered')

    @property
    @abc.abstractmethod
    def _Parameters(self):
        '''Parameter class for this model.'''

    def __init__(self, **kwds):
        self.parameters = self._Parameters(**kwds)
