'''Recovery from infectious to recovered.'''

from ._utility import waiting_time


class Recovery(waiting_time.Gamma):
    '''Recovery from infectious to recovered.'''

    def __init__(self, parameters):
        super().__init__(parameters.recovery_mean,
                         parameters.recovery_shape)
