'''Automatic differentiation.'''

import functools

import numpy
import torch.autograd.functional


def jacobian(func, vectorize=True):
    '''For `func(t, y)`, get the Jacobian matrix with respect to `y`
    at time `t`.'''
    def jac(t, y):
        func_y = functools.partial(func, t)
        y_tensor = torch.tensor(y)
        J = torch.autograd.functional.jacobian(func_y, y_tensor,
                                               vectorize=vectorize)
        return numpy.stack(J)
    return jac
