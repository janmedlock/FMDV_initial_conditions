'''Get the fundamental solution.'''

import numpy

from .. import _utility


class _Solver:
    '''Crank–Nicolson solver for the variational equation.'''

    def __init__(self, model, y):
        self.model = model
        self.y = y
        self._sparse = self.model._solver._sparse
        self._build_matrices()

    def _build_matrices(self):
        '''Build matrices needed by the solver.'''
        self.I = self._I()

    def _I(self):
        '''Build the identity matrix.'''
        n = self.y.shape[-1]
        I = _utility.numerical.identity(n, sparse=self._sparse)
        return I

    def jacobian(self, t_cur):
        '''Get the Jacobian at (t, y(t)).'''
        i = self.y.index.get_loc(t_cur)
        y_cur = self.y.iloc[i]
        y_new = self.y.iloc[i + 1]
        return self.model._solver.jacobian(t_cur, y_cur, y_new)

    def step(self, t_cur, Phi_cur, t_new, display=False):
        '''Do a step.'''
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
        return _utility.linalg.solve(IJ_new, IJPhi_cur,
                                     overwrite_a=True,
                                     overwrite_b=True)

    def solve(self, display=False):
        '''Solve.'''
        t = self.y.index
        Phi = numpy.empty((len(t), ) + self.I.shape)
        if self._sparse:
            Phi[0] = self.I.toarray()
        else:
            Phi[0] = self.I
        for k in range(1, len(t)):
            Phi[k] = self.step(t[k - 1], Phi[k - 1], t[k], display=display)
        return Phi


def solution(model, y, display=False):
    '''Solve for the fundamental solution.'''
    solver = _Solver(model, y)
    return solver.solve(display=display)