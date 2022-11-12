'''Automatic differentiation.'''

import functools

import numpy

import torch.autograd.functional


def jacobian(func):
    '''Get the Jacobian matrix for the vector-valued `func`.'''
    def jac(t, y):
        return numpy.stack(
            torch.autograd.functional.jacobian(
                functools.partial(func, t),
                torch.tensor(y),
                vectorize=True
            )
        )
    return jac
