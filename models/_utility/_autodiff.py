'''Automatic differentiation.'''

import functools

import torch


def jacobian(func, vectorize=True):
    '''For `func(t, y)`, get the Jacobian matrix with respect to `y`
    at time `t`.'''
    def func_jac(t, y):
        func_y = functools.partial(func, t)
        y_tensor = torch.as_tensor(y)
        jac_conj = torch.autograd.functional.jacobian(func_y,
                                                      y_tensor,
                                                      vectorize=vectorize)
        if isinstance(jac_conj, tuple):
            jac_conj = torch.stack(jac_conj)
        jac = jac_conj.conj()
        return jac.numpy(force=True)
    return func_jac
