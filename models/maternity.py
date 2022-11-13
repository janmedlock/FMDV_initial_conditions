'''Maternity.'''

import numpy


class Maternity:
    '''Maternity.'''

    _age_menarchy = 4  # years

    def __init__(self, parameters):
        # Maternity does not depend on `parameters`.
        pass

    def __call__(self, age):
        '''Maternity.'''
        # 0 before menarchy and 1 after.
        return numpy.where(age < self._age_menarchy, 0, 1)
