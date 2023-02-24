'''Progression for exposed to infectious.'''

from ._utility import waiting_time


class Progression(waiting_time.Gamma):
    '''Progression for exposed to infectious.'''

    def __init__(self, parameters):
        super().__init__(parameters.progression_mean,
                         parameters.progression_shape)
