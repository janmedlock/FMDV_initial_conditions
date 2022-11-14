'''Get the fundamental solution.'''

import numpy
import scipy.linalg

from .. import _utility


class _VariationalSolver:
    '''Crank–Nicolson solver for the variational equation.'''

    def __init__(self, func, y):
        self._jacobian = _utility.jacobian(func)
        self.y = y

    def jacobian(self, t):
        '''The Jacobian at (t, y(t)).'''
        return self._jacobian(t, self.y.loc[t])

    def step(self, t_cur, phi_cur, t_new, phi_new):
        '''Crank–Nicolson step.'''
        t_step = t_new - t_cur
        # self.B[:] = (self.eye + t_step / 2 * self.jacobian(t_cur)) @ phi_cur
        # As above, but avoid building new arrays.
        numpy.multiply(t_step / 2, self.jacobian_val, out=self.B)
        numpy.add(self.eye, self.B, out=self.temp)
        numpy.dot(self.temp, phi_cur, out=self.B)
        # self.A[:] = self.eye - t_step / 2 * self.jacobian(t_new)
        # As above, but avoid building new arrays.
        # `self.jacobian_val` will get used in the next call of `_step()`.
        self.jacobian_val = self.jacobian(t_new)
        numpy.multiply(- t_step / 2, self.jacobian_val, out=self.temp)
        numpy.add(self.eye, self.temp, out=self.A)
        # Solve self.A @ phi_new = self.B for phi_new.
        phi_new[:] = scipy.linalg.solve(self.A, self.B,
                                        overwrite_a=True,
                                        overwrite_b=True)

    def solve(self):
        '''Solve.'''
        t = self.y.index
        n = self.y.shape[-1]
        phi = numpy.empty((len(t), n, n))
        phi[0] = self.eye = numpy.eye(n)
        # Initialize temporary storage used in `step()`.
        self.A = numpy.empty((n, n))
        self.B = numpy.empty((n, n))
        self.temp = numpy.empty((n, n))
        self.jacobian_val = self.jacobian(t[0])
        for k in range(1, len(t)):
            self.step(t[k - 1], phi[k - 1], t[k], phi[k])
        return phi


def solution(func, y, **kwds):
    '''Solve for the fundamental solution.'''
    return _VariationalSolver(func, y).solve()
