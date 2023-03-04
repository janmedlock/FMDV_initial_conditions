'''Transmission.'''


class Transmission:
    '''Transmission.'''

    def __init__(self, parameters):
        self.rate = parameters.transmission_rate
        assert self.rate >= 0
