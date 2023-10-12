'''Automatic differentiation.'''

import functools

import torch


def jacobian(func, vectorize=True):
    '''For `func(t, y)`, get the Jacobian matrix with respect to `y`
    at time `t`.'''
    def jac(t, y):
        func_y = functools.partial(func, t)
        y_tensor = torch.as_tensor(y)
        J_conj = torch.autograd.functional.jacobian(func_y,
                                                    y_tensor,
                                                    vectorize=vectorize)
        if isinstance(J_conj, tuple):
            J_conj = torch.stack(J_conj)
        J = J_conj.conj()
        return J.numpy(force=True)
    return jac
