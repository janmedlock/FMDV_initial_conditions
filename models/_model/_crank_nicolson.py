'''Crank–Nicolson Mixin class.'''

import numpy


class Mixin:  # pylint: disable=too-few-public-methods
    '''Crank–Nicolson helper mixin.'''

    _q_vals = ('new', 'cur')

    def _cn_op(self, q, X, Y, out=None):  # pylint: disable=invalid-name
        '''X ± t_step / 2 * Y.'''
        # pylint: disable=invalid-name
        scaled_Y = numpy.multiply(self.t_step / 2, Y, out=out)
        if q == 'new':
            op = numpy.subtract
        elif q == 'cur':
            op = numpy.add
        else:
            raise ValueError(f'{q=}')
        return op(X, scaled_Y, out=out)
