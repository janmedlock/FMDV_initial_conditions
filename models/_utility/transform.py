'''Transformations.'''

import numpy


class Identity:
    '''`y` -> `y`.'''

    def __init__(self, y):
        pass

    @staticmethod
    def __call__(y):
        return y

    @staticmethod
    def inverse(y):
        return y


class ConstantSum:
    '''Reduce the dimension of `y` by 1 using its sum.'''

    def __init__(self, y):
        self.y_sum = y.sum()

    @staticmethod
    def __call__(y):
        '''Reduce the dimension of `y`.'''
        return y[:-1]

    def inverse(self, x):
        '''Expand the dimension of `x`.'''
        return numpy.hstack((x, self.y_sum - x.sum()))
