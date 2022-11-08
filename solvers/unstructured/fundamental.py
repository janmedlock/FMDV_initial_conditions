'''Get the fundamental solution.'''

import numpy

from . import solver


class _VariationalEquation:
    def __init__(self, jacobian, y):
        self.jacobian = jacobian
        self.y = y

    def __call__(self, t, phi_raveled):
        phi = phi_raveled.reshape((self.y.shape[-1], -1))
        d_phi = self.jacobian(t, self.y.loc[t]) @ phi
        return d_phi.ravel()


def solution(jacobian, y, **kwds):
    '''Solve for the fundamental solution.'''
    var_eq = _VariationalEquation(jacobian, y)
    solver_ = solver.Solver.create(var_eq, **kwds)
    t = y.index
    phi_0 = numpy.eye(y.shape[-1])
    phi_raveled = solver_(t, phi_0.ravel(), _solution=False)
    return phi_raveled.reshape((len(t), *phi_0.shape))
