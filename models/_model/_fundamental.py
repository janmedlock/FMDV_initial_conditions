'''Get the fundamental solution.'''

import functools

import numpy

from . import solver
from .. import _utility


class _Solver(solver.Base):
    '''Crank–Nicolson solver for the variational equation.'''

    @property
    def sparse(self):
        '''Whether the solver uses sparse arrays and sparse linear
        algebra.'''
        # pylint: disable-next=protected-access
        return self.model._solver.sparse

    @functools.cached_property
    def I(self):  # pylint: disable=invalid-name  # noqa: E743
        '''The identity matrix.'''
        return _utility.numerical.identity(self.y.shape[-1],
                                           sparse=self.sparse)

    def __init__(self, model, y):
        self.model = model
        self.y = y
        self.t_step = 0.  # This is updated in `.step()`.

    def jacobian(self, t_cur):
        '''Get the Jacobian at (t, y(t)).'''
        i = self.y.index.get_loc(t_cur)
        y_cur = self.y.iloc[i]
        y_new = self.y.iloc[i + 1]
        # pylint: disable-next=protected-access
        return self.model._solver.jacobian(t_cur, y_cur, y_new)

    # pylint: disable-next=invalid-name
    def step(self, t_cur, Phi_cur, t_new, display=False):
        '''Do a step.'''
        if display:
            print(f'{t_new=}')
        # The Crank–Nicolson scheme is
        # (Phi_new - Phi_cur) / t_step
        # = J(t_cur) @ (Phi_new + Phi_cur) / 2,
        # or
        # (I - t_step / 2 * J(t_cur)) @ Phi_new
        # = (I + t_step / 2 * J(t_cur)) @ Phi_cur.
        self.t_step = t_new - t_cur
        J = self.jacobian(t_cur)  # pylint: disable=invalid-name
        IJ_cur = self._cn_op('cur', self.I, J)  # pylint: disable=invalid-name
        IJPhi_cur = IJ_cur @ Phi_cur  # pylint: disable=invalid-name
        IJ_new = self._cn_op('new', self.I, J)  # pylint: disable=invalid-name
        return _utility.linalg.solve(IJ_new, IJPhi_cur,
                                     overwrite_a=True,
                                     overwrite_b=True)

    @property
    def _I_dense(self):  # pylint: disable=invalid-name
        '''Return a dense identity matrix.'''
        if self.sparse:
            val = self.I.toarray()
        else:
            val = self.I
        return val

    def monodromy(self, display=False):
        '''Solve for the monodromy matrix.'''
        t = self.y.index
        # pylint: disable-next=invalid-name
        Phi_temp = numpy.empty((2, ) + self.I.shape)
        (Phi_cur, Phi_new) = Phi_temp  # pylint: disable=invalid-name
        Phi_new[:] = self._I_dense
        for k in range(1, len(t)):
            # Update so that what was the new value of the solution is
            # now the current value and what was the current value of
            # the solution will be storage space for the new value.
            # pylint: disable-next=invalid-name
            (Phi_cur, Phi_new) = (Phi_new, Phi_cur)
            Phi_new[:] = self.step(t[k - 1], Phi_cur, t[k], display=display)
        return Phi_new

    def solve(self, display=False):
        '''Solve.'''
        t = self.y.index
        # pylint: disable=invalid-name
        Phi = numpy.empty((len(t), ) + self.I.shape)
        Phi[0] = self._I_dense
        for k in range(1, len(t)):
            Phi[k] = self.step(t[k - 1], Phi[k - 1], t[k], display=display)
        return Phi


def monodromy(model, y, display=False):
    '''Solve for the monodromy matrix.'''
    solver_ = _Solver(model, y)
    return solver_.monodromy(display=display)


def solution(model, y, display=False):
    '''Solve for the fundamental solution.'''
    solver_ = _Solver(model, y)
    return solver_.solve(display=display)
