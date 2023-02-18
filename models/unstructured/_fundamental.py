'''Get the fundamental solution.'''

import numpy
import scipy.linalg


class _Solver:
    '''Crank–Nicolson solver for the variational equation.'''

    def __init__(self, model, y):
        self.model = model
        self.y = y
        self.eye = numpy.eye(self.y.shape[-1])

    def jacobian(self, t):
        '''The Jacobian at (t, y(t)).'''
        return self.model.jacobian(t, self.y.loc[t])

    def step(self, t_cur, Phi_cur, t_new):
        '''Crank–Nicolson step.'''
        # The Crank–Nicolson scheme is
        # (Phi_new - Phi_cur) / t_step
        # = (J(t_new) @ Phi_new + J(t_cur) @ Phi_cur) / 2.
        # Define
        # IJ0 = I - t_step / 2 * J(t_new),
        # and
        # IJPhi1 = [I + t_step / 2 * J(t_cur)] @ Phi_cur,
        # so that
        # IJ0 @ Phi_new = IJPhi1.
        t_step = t_new - t_cur
        IJPhi1 = (self.eye + t_step / 2 * self.jacobian(t_cur)) @ Phi_cur
        IJ0 = self.eye - t_step / 2 * self.jacobian(t_new)
        return scipy.linalg.solve(IJ0, IJPhi1,
                                  overwrite_a=True,
                                  overwrite_b=True)

    def solve(self):
        '''Solve.'''
        t = self.y.index
        Phi = numpy.empty((len(t), ) + self.eye.shape)
        Phi[0] = self.eye
        for k in range(1, len(t)):
            Phi[k] = self.step(t[k - 1], Phi[k - 1], t[k])
        return Phi


def solution(model, y):
    '''Solve for the fundamental solution.'''
    solver = _Solver(model, y)
    return solver.solve()
