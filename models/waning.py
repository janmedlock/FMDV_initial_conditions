'''Waning of maternal immunity.'''

from . import _utility


class Waning(_utility.WaitingTimeGamma):
    '''Waning of maternal immunity.'''

    def __init__(self, parameters):
        super().__init__(parameters.maternal_immunity_mean,
                         parameters.maternal_immunity_shape)
