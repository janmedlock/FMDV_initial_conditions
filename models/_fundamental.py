'''Get the fundamental solution.'''

import numpy
import scipy.linalg


class _Solver:
    '''Crank–Nicolson solver for the variational equation.'''

    def __init__(self, model, y):
        self.model = model
        self.y = y
        self.I = numpy.identity(self.y.shape[-1])

    def jacobian(self, t_cur):
        '''The Jacobian at (t, y(t)).'''
        i = self.y.index.get_loc(t_cur)
        y_cur = self.y.iloc[i]
        y_new = self.y.iloc[i + 1]
        return self.model._solver.jacobian(t_cur, y_cur, y_new)

    def step(self, t_cur, Phi_cur, t_new, display=False):
        '''Crank–Nicolson step.'''
        if display:
            print(f'{t_new=}')
        # The Crank–Nicolson scheme is
        # (Phi_new - Phi_cur) / t_step
        # = J(t_cur) @ (Phi_new + Phi_cur) / 2.
        t_step = t_new - t_cur
        J = self.jacobian(t_cur)
        IJ_new = self.I - t_step / 2 * J
        IJ_cur = self.I + t_step / 2 * J
        IJPhi_cur = IJ_cur @ Phi_cur
        return scipy.linalg.solve(IJ_new, IJPhi_cur,
                                  overwrite_a=True,
                                  overwrite_b=True)

    def solve(self, display=False):
        '''Solve.'''
        t = self.y.index
        Phi = numpy.empty((len(t), ) + self.I.shape)
        Phi[0] = self.I
        for k in range(1, len(t)):
            Phi[k] = self.step(t[k - 1], Phi[k - 1], t[k], display=display)
        return Phi


def solution(model, y, display=False):
    '''Solve for the fundamental solution.'''
    solver = _Solver(model, y)
    return solver.solve(display=display)
