'''Recovery from infectious to recovered.'''

from . import _utility


class Recovery(_utility.WaitingTimeGamma):
    '''Recovery from infectious to recovered.'''

    def __init__(self, parameters):
        super().__init__(parameters.recovery_mean,
                         parameters.recovery_shape)
