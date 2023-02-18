'''Get the fundamental solution.'''

import numpy
import scipy.linalg


class _VariationalSolver:
    '''Crank–Nicolson solver for the variational equation.'''

    def __init__(self, model, y):
        self.model = model
        self.y = y

    def jacobian(self, t):
        '''The Jacobian at (t, y(t)).'''
        return self.model.jacobian(t, self.y.loc[t])

    def step(self, t_cur, phi_cur, t_new, phi_new):
        '''Crank–Nicolson step.'''
        # The Crank–Nicolson scheme is
        # (phi_new - phi_cur) / t_step
        # = (J(t_new) @ phi_new + J(t_cur) @ phi_cur) / 2.
        # Define
        # IJ0 = I - t_step / 2 * J(t_new),
        # and
        # IJphi1 = [I + t_step / 2 * J(t_cur)] @ phi_cur,
        # so that
        # IJ0 @ phi_new = IJphi1.
        t_step = t_new - t_cur
        # `self.jacobian_val = self.jacobian(t_cur)` from the previous
        # call of `_step()` (or from initialization in `solve()` in
        # the first call of `_step()`).
        self.IJphi1[:] = (self.eye + t_step / 2 * self.jacobian_val) @ phi_cur
        # `self.jacobian_val` will get used again in the next call of
        # `_step()`.
        self.jacobian_val = self.jacobian(t_new)
        self.IJ0[:] = self.eye - t_step / 2 * self.jacobian_val
        phi_new[:] = scipy.linalg.solve(self.IJ0, self.IJphi1,
                                        overwrite_a=True,
                                        overwrite_b=True)

    def solve(self):
        '''Solve.'''
        t = self.y.index
        n = self.y.shape[-1]
        phi = numpy.empty((len(t), n, n))
        phi[0] = self.eye = numpy.eye(n)
        # Initialize temporary storage used in `step()`.
        self.IJ0 = numpy.empty((n, n))
        self.IJphi1 = numpy.empty((n, n))
        self.jacobian_val = self.jacobian(t[0])
        for k in range(1, len(t)):
            self.step(t[k - 1], phi[k - 1], t[k], phi[k])
        return phi


def solution(model, y):
    '''Solve for the fundamental solution.'''
    return _VariationalSolver(model, y).solve()
