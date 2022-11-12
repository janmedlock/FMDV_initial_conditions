'''Maternity rate.'''

import numpy


class Maternity:
    '''Maternity rate.'''

    @staticmethod
    def maternity_rate(age):
        '''Maternity rate.'''
        # 0 for ages less than 4 years and
        # 1 for the rest.
        return numpy.where(age < 4, 0, 1)
