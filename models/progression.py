'''Progression for exposed to infectious.'''

from . import _utility


class Progression(_utility.waiting_time.Gamma):
    '''Progression for exposed to infectious.'''

    def __init__(self, parameters):
        super().__init__(parameters.progression_mean,
                         parameters.progression_shape)
